"""
src/timesfm_forecaster.py

A per-series, zero-shot forecaster that turns Google's pretrained TimesFM
foundation model into a full-coverage Kaggle submission. Like
`src/arima_forecaster.py`'s `ArimaForecaster`, it is deliberately NOT
sklearn-compatible - it exposes the plain `.fit(raw_train) -> self` /
`.predict(raw_test) -> np.ndarray` contract that `CLAUDE.md` and
`docs/person_a_prompt.md` allow for the non-sklearn model tracks ("a plain
class with .predict(raw_test_df) is fine here").

Strategy (zero-shot, no training, no feature engineering):
  For each (Store, Dept) series, feed its past weekly `Weekly_Sales` history
  (the last `context_len` weeks) to TimesFM and ask for a `horizon_len`-step
  forecast, then align that forecast to the series' test dates. Clipped at 0.

Why this shape:
  - TimesFM is a pretrained time-series foundation model: it forecasts from raw
    history with no per-series fitting. The whole point of this bonus experiment
    is to see how a zero-shot pretrained model does on this task WITHOUT the
    lag/rolling/aggregate feature machinery the tree models rely on.
  - Context = the full available history (up to ~143 weeks for a full-length
    series). That comfortably spans more than one 52-week seasonal cycle, so the
    model can pick up the yearly retail seasonality on its own. Test weeks are
    contiguous immediately after train's end, so a single forward forecast lines
    up with the test dates by position.
  - The far end of a ~39-week horizon is expected to be the weak part (a long
    horizon relative to typical foundation-model context) - documented, not fixed.

Performance:
  Inference is a single batched TimesFM call over all eligible series, not a
  per-series fit. It benefits from a GPU - run on a Colab GPU runtime. On CPU it
  still works but is slow. There is no per-series parallelism to manage here (the
  batching lives inside TimesFM), unlike the ARIMA forecaster's joblib fan-out.

Fallbacks (guarantee full coverage, never NaN) - same cascade as ArimaForecaster:
  - Series shorter than `min_context` weeks, or a series that produced no usable
    context -> that series' own mean (`store_dept_mean_`).
  - A (Store, Dept) pair unseen in train (new department in test) -> store mean,
    then global mean.
  - Any residual NaN after assembly -> global mean. Everything clipped at 0.

`timesfm` is imported lazily inside `_load_model` so that importing this module
never fails in an environment where timesfm is not installed - the class stays
importable (and picklable) regardless. The loaded model is excluded from pickling
(`__getstate__`) so `joblib.dump(forecaster, ...)` stores only the cached history
and config; the weights come from the Hugging Face checkpoint id, not the pickle.

The exact TimesFM API call is isolated in `_forecast_batch` - the `timesfm`
package's API has changed across versions, so that one method is the single place
to adjust if the installed version differs.

No em dashes anywhere (project code-style rule) - hyphens only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TimesFMForecaster:
    """Per-series zero-shot TimesFM forecaster for raw test frames.

    Targets the TimesFM 2.5 package API
    (`timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)` + `model.compile(
    timesfm.ForecastConfig(...))` + `model.forecast(horizon=..., inputs=[...])`).
    This is the version that installs on current Colab (Python 3.12); the older
    1.x `TimesFm`/`TimesFmHparams` API is not used.

    Parameters
    ----------
    m:
        Seasonal period in weeks (52 for weekly data with yearly seasonality).
        Used only to size the weekly reindex grid / sanity messaging; TimesFM
        infers seasonality from the context itself.
    context_len:
        Number of most-recent weekly observations fed to TimesFM as context, and
        the model's `max_context`. Should be >= the longest per-series history
        (~143 weeks) so the full history is used. TimesFM 2.5 supports very long
        contexts; 512 is ample here (more than one 52-week cycle) and fast.
    horizon_len:
        Number of steps TimesFM forecasts ahead. 39 for the full Kaggle test span
        (2012-11-02 -> 2013-07-26); pass 13 when validating on the shared 13-week
        holdout fold.
    max_horizon:
        `max_horizon` passed to `ForecastConfig` at compile time. The effective
        value is `max(horizon_len, max_horizon)`, so the compiled model always
        covers the requested horizon.
    checkpoint:
        Hugging Face repo id of the TimesFM 2.5 checkpoint to load.
    min_context:
        Minimum number of weekly observations a series needs before TimesFM is
        trusted. Shorter series fall back to their own mean (mirrors ARIMA's
        `min_history`).
    infer_batch_size:
        Number of series per `model.forecast` call. Series are chunked at this
        size and the results concatenated, to bound GPU memory across the ~3,000
        series in the full panel.
    normalize_inputs, infer_is_positive:
        `ForecastConfig` flags. `normalize_inputs` per-series-normalizes the
        context (helps across the wildly different sales scales); `infer_is_positive`
        keeps forecasts non-negative, which suits sales (we still clip at 0 too).
    verbose:
        If True, print how many series went to TimesFM vs the fallback.
    """

    def __init__(
        self,
        m: int = 52,
        context_len: int = 512,
        horizon_len: int = 39,
        max_horizon: int = 64,
        checkpoint: str = "google/timesfm-2.5-200m-pytorch",
        min_context: int = 52,
        infer_batch_size: int = 512,
        normalize_inputs: bool = True,
        infer_is_positive: bool = True,
        verbose: bool = True,
    ):
        self.m = m
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.max_horizon = max_horizon
        self.checkpoint = checkpoint
        self.min_context = min_context
        self.infer_batch_size = infer_batch_size
        self.normalize_inputs = normalize_inputs
        self.infer_is_positive = infer_is_positive
        self.verbose = verbose
        self._model = None  # loaded lazily, never pickled

    # ------------------------------------------------------------------ fit
    def fit(self, raw_train: pd.DataFrame) -> "TimesFMForecaster":
        """Cache per-(Store, Dept) weekly history and fallback aggregates.

        `raw_train` is a raw `train.csv`-shaped frame - only Store, Dept, Date,
        and Weekly_Sales are needed. TimesFM is zero-shot, so nothing is trained
        here; the model is loaded lazily at predict time."""
        required = {"Store", "Dept", "Date", "Weekly_Sales"}
        missing = required - set(raw_train.columns)
        if missing:
            raise ValueError(f"fit() requires columns {sorted(required)}, missing {sorted(missing)}")

        df = raw_train[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        self.global_mean_ = float(df["Weekly_Sales"].mean())
        self.store_mean_ = df.groupby("Store")["Weekly_Sales"].mean().to_dict()

        self.history_: dict[tuple[int, int], pd.Series] = {}
        self.store_dept_mean_: dict[tuple[int, int], float] = {}
        for (store, dept), g in df.groupby(["Store", "Dept"], sort=False):
            ser = g.set_index("Date")["Weekly_Sales"].sort_index()
            # collapse any duplicate dates (keep the last observation for that week)
            ser = ser[~ser.index.duplicated(keep="last")]
            self.history_[(int(store), int(dept))] = ser
            self.store_dept_mean_[(int(store), int(dept))] = float(ser.mean())

        return self

    # -------------------------------------------------------------- predict
    def predict(self, raw_test: pd.DataFrame) -> np.ndarray:
        """Forecast Weekly_Sales for every row of a raw `test.csv`-shaped frame.

        Returns a finite float array aligned to `raw_test`'s row order."""
        if not hasattr(self, "history_"):
            raise RuntimeError("TimesFMForecaster must be fit() before predict().")

        required = {"Store", "Dept", "Date"}
        missing = required - set(raw_test.columns)
        if missing:
            raise ValueError(f"predict() requires columns {sorted(required)}, missing {sorted(missing)}")

        df = raw_test[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Pass 1 (main process): per series, build the TimesFM context array and
        # a level fallback. Queue eligible series for the batched inference call.
        plans = []          # one dict per (Store, Dept) group
        inputs = []         # context arrays for series that go to TimesFM
        for (store, dept), g in df.groupby(["Store", "Dept"], sort=False):
            test_dates = sorted(pd.unique(g["Date"]))
            context, level = self._prepare_series(int(store), int(dept))
            plan = {
                "row_index": g.index,
                "row_dates": g["Date"].to_numpy(),
                "dates": test_dates,
                "level": level,
                "job": None,
            }
            if context is not None:
                plan["job"] = len(inputs)
                inputs.append(context)
            plans.append(plan)

        # Pass 2: one batched zero-shot TimesFM call over all eligible series.
        if inputs:
            if self.verbose:
                print(f"Forecasting {len(inputs)} series with TimesFM "
                      f"(horizon={self.horizon_len}, context<={self.context_len}) ...")
            forecasts = self._forecast_batch(inputs)
            if self.verbose:
                n_fallback = len(plans) - len(inputs)
                print(f"  TimesFM forecast done. {len(inputs)} series via TimesFM, "
                      f"{n_fallback} via mean fallback (too short / unseen).")
        else:
            forecasts = []

        # Pass 3: assemble per-series forecasts, map back to test-row order.
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for plan in plans:
            n = len(plan["dates"])
            if plan["job"] is not None:
                fc = np.asarray(forecasts[plan["job"]], dtype=float)
                # TimesFM returns horizon_len steps; take the first n and, if the
                # series needs more than we forecast, pad with the last value.
                if fc.shape[0] >= n:
                    preds_sorted = fc[:n]
                else:
                    pad = np.full(n - fc.shape[0], fc[-1] if fc.shape[0] else plan["level"])
                    preds_sorted = np.concatenate([fc, pad])
                # Any non-finite step falls back to the series level.
                preds_sorted = np.where(np.isfinite(preds_sorted), preds_sorted, plan["level"])
            else:
                preds_sorted = np.full(n, plan["level"], dtype=float)

            preds_sorted = np.clip(preds_sorted, 0.0, None)
            date_to_pred = dict(zip(plan["dates"], preds_sorted))
            out.loc[plan["row_index"]] = pd.Series(plan["row_dates"]).map(date_to_pred).to_numpy()

        # Safety net: nothing should be NaN, but guarantee it for the submission.
        out = out.fillna(self.global_mean_).clip(lower=0.0)
        return out.to_numpy()

    # ---------------------------------------------------------- internals
    def _prepare_series(self, store: int, dept: int):
        """Return (context_array_or_None, level) for one series.

        `context_array` is the last `context_len` weeks of the series on a regular
        weekly grid (small internal gaps filled), or None when the series is
        unseen / shorter than `min_context` (caller then uses `level` alone).
        `level` is the fallback constant: store_dept mean -> store mean -> global."""
        ser = self.history_.get((store, dept))

        level = self.store_dept_mean_.get((store, dept))
        if level is None or not np.isfinite(level):
            level = self.store_mean_.get(store, self.global_mean_)
        if not np.isfinite(level):
            level = self.global_mean_
        level = max(0.0, float(level))

        if ser is None or len(ser) < self.min_context:
            return None, level

        # Reindex onto a regular weekly grid so TimesFM sees a contiguous series,
        # filling small internal gaps (interpolate, then edge-fill any residual).
        full_idx = pd.date_range(ser.index.min(), ser.index.max(), freq="7D")
        grid = ser.reindex(full_idx)
        if grid.isna().any():
            grid = grid.interpolate(limit_direction="both")
            grid = grid.ffill().bfill()

        values = grid.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or len(values) < self.min_context:
            return None, level

        context = values[-self.context_len:]
        return context, level

    def _load_model(self):
        """Lazily import, construct, and compile the TimesFM 2.5 model. Isolated
        so the module stays importable without timesfm installed, and so the model
        is built at most once per process."""
        if self._model is not None:
            return self._model

        import timesfm

        # This wrapper targets the timesfm 2.5 API. Fail loudly and actionably if
        # the installed package exposes a different interface, rather than with a
        # deep AttributeError inside the constructor.
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError(
                "The installed `timesfm` does not expose the 2.5 API "
                "(`timesfm.TimesFM_2p5_200M_torch`) this wrapper targets. Install "
                "'timesfm[torch]' and, if timesfm was already imported this session, "
                "restart the runtime and re-run from the top."
            )

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=max(self.horizon_len, self.max_horizon),
                normalize_inputs=self.normalize_inputs,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=self.infer_is_positive,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        return self._model

    def _forecast_batch(self, inputs: list) -> list:
        """Run the TimesFM 2.5 zero-shot forecast over `inputs` (a list of 1D
        context arrays) and return a list of `horizon_len`-step point forecasts.
        Series are processed in chunks of `infer_batch_size` to bound GPU memory.

        This is the single place that touches the `timesfm` package's forecast
        API - adjust here if the installed version differs."""
        model = self._load_model()
        out = []
        for start in range(0, len(inputs), self.infer_batch_size):
            chunk = [np.asarray(x, dtype=float) for x in inputs[start:start + self.infer_batch_size]]
            point_forecast, _ = model.forecast(horizon=self.horizon_len, inputs=chunk)
            out.extend(np.asarray(row, dtype=float) for row in point_forecast)
        return out

    # ------------------------------------------------------------ pickling
    def __getstate__(self):
        """Exclude the loaded TimesFM model from pickling - only the cached
        history and config are persisted; weights come from the HF checkpoint."""
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None
