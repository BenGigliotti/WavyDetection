# te_qc_gui.py
# Wavy Detection — Prototype Dashboard with History/Trend + Data-driven Gauge
# Built only with Python stdlib + numpy/pandas/matplotlib/sklearn.
# Integrated waviness class overlays (from Streamlit app) for History & Results pages.

import os, csv, math, random, subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import numpy as np, pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

APP_TITLE   = "Wavy Detection Prototype Dashboard"
APP_VERSION = "v0.4 (history+needle+class overlays)"

# ---- Force the OD column by exact name ----
EXACT_OD_COLUMN = "Tag_value"  # <- put your exact Excel header here
ENFORCE_EXACT_OD = True        # keep True until it plots correctly




DATA_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime", "t_stamp"],
    "od":   ["od", "outer_diameter", "tube_od", "ndc_od_value",
             "ndc_system_ovality_value__tag_value", "tag_value"],
}


def rolling_mad(x):
    m = np.median(x)
    return np.mean(np.abs(x - m))

# ---- Wavy class palette (matches your Streamlit app) ----
CLASS_COLORS = {
    "STEADY":        "#4CAF50",
    "MILD_WAVE":     "#FFB300",
    "STRONG_WAVE":   "#E53935",
    "DRIFT":         "#2196F3",
    "BURSTY_NOISY":  "#8E24AA",
    "UNCERTAIN":     "#9E9E9E",
}

def pastel(hex_color: str, alpha: float = 0.25) -> str:
    """Blend an RGB hex color with white by `alpha` (0..1)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # blend toward white
    r = int((1 - alpha) * 255 + alpha * r)
    g = int((1 - alpha) * 255 + alpha * g)
    b = int((1 - alpha) * 255 + alpha * b)
    return f"#{r:02X}{g:02X}{b:02X}"


def normalize_label(s) -> str:
    """
    Map free-text/ID labels to canonical class keys used for coloring.
    Returns one of: STEADY, MILD_WAVE, STRONG_WAVE, DRIFT, BURSTY_NOISY, UNCERTAIN
    """
    if s is None:
        return "UNCERTAIN"

    # Numeric IDs (0..5) and numeric-like strings
    try:
        v = int(float(str(s).strip()))
        id_map = {0: "STEADY", 1: "MILD_WAVE", 2: "STRONG_WAVE", 3: "DRIFT", 4: "BURSTY_NOISY", 5: "UNCERTAIN"}
        return id_map.get(v, "UNCERTAIN")
    except Exception:
        pass

    t = str(s).strip().upper()
    t = t.replace("-", "_").replace(" ", "_").replace("/", "_")

    aliases = {
        # steady / good
        "STEADY": "STEADY", "OK": "STEADY", "GOOD": "STEADY", "NORMAL": "STEADY", "FLAT": "STEADY",
        "STABLE": "STEADY", "STEADY_STATE": "STEADY",

        # mild wave
        "MILD": "MILD_WAVE", "MILD_WAVE": "MILD_WAVE", "LIGHT_WAVE": "MILD_WAVE",
        "SMALL_WAVE": "MILD_WAVE", "MINOR_WAVE": "MILD_WAVE",

        # strong wave / chatter
        "STRONG": "STRONG_WAVE", "STRONG_WAVE": "STRONG_WAVE",
        "WAVY": "STRONG_WAVE", "CHATTER": "STRONG_WAVE", "HEAVY_WAVE": "STRONG_WAVE",

        # drift / trend
        "DRIFT": "DRIFT", "DRIFTING": "DRIFT", "TREND": "DRIFT", "SLOPE": "DRIFT",

        # noisy / burst
        "BURSTY_NOISY": "BURSTY_NOISY", "BURSTY_NOISE": "BURSTY_NOISY",
        "NOISY": "BURSTY_NOISY", "BURST": "BURSTY_NOISY", "SPIKY": "BURSTY_NOISY",

        # unknown
        "UNCERTAIN": "UNCERTAIN", "UNKNOWN": "UNCERTAIN",
    }
    return aliases.get(t, "UNCERTAIN")


# Optional: quick class visibility hook (no UI yet; modify as needed)
VISIBLE_CLASSES = set(CLASS_COLORS.keys())

# ========================= Utilities =========================
def open_in_vscode(path="."):
    try:
        subprocess.Popen(["code", path]); return
    except Exception:
        pass
    for c in [
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        r"C:\Program Files\Microsoft VS Code\Code.exe",
        r"C:\Program Files (x86)\Microsoft VS Code\Code.exe",
    ]:
        exe = os.path.expandvars(c)
        if os.path.exists(exe):
            subprocess.Popen([exe, path]); return

def pick(colnames, candidates):
    low = [c.lower() for c in colnames]
    for alias in candidates:
        if alias.lower() in low:
            return colnames[low.index(alias.lower())]
    return None

def try_float(x):
    try: return float(x)
    except: return None

# ========================= Data Store =========================
class DataStore:
    # ---- Global data store (must exist before pages use it) ----

    def __init__(self):
        self.path = None
        self.ts = []       # list[str] raw ts strings
        self.ts_dt = []    # list[pd.Timestamp] parsed timestamps aligned with self.od
        self.od = []       # list[float]
        self.classes = []  # list[dict]: {"start":ts, "end":ts, "label":str, "i0":int, "i1":int}
        self.last_loaded_rows = 0

    def _read_any_table(self, path: str):
        """Return a pandas DataFrame from CSV/XLS/XLSX (first sheet for Excel)."""
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path, sheet_name=0)
        else:
            return pd.read_csv(path)
        
    def _smart_to_numeric(self, series: pd.Series) -> pd.Series:
        """
        Convert numbers that may be stored as strings with units, commas, or thousands
        separators into float. Returns a float Series; non-parsable values -> NaN.
        """
        s = series.copy()

        # If it's already numeric, return as is
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")

        # Make strings
        s = s.astype(str).str.strip()

        # Remove common unit suffixes and any non-numeric decorations except . , - and exponent
        # Example: "1.63 mm" -> "1.63", "1,63mm" -> "1,63"
        s = s.str.replace(r"[^\d\.,\-eE+]", "", regex=True)

        # Heuristic: detect decimal comma vs decimal dot
        has_comma = s.str.contains(",", regex=False, na=False).sum()
        has_dot   = s.str.contains(".", regex=False, na=False).sum()

        # If comma appears more commonly than dot, treat comma as decimal separator.
        # Remove thousand separators accordingly.
        if has_comma > has_dot:
            # First remove dot thousands separators if any (e.g., "1.234,56" -> "1234,56")
            s = s.str.replace(".", "", regex=False)
            # Then convert decimal comma to dot
            s = s.str.replace(",", ".", regex=False)
        else:
            # Treat dot as decimal separator; remove comma thousands separators
            # e.g., "1,234.56" -> "1234.56"
            # (Avoid breaking "1,23" if it was decimal comma but rarer than dot overall)
            s = s.str.replace(",", "", regex=False)

        # Final conversion
        return pd.to_numeric(s, errors="coerce")

    def _try_autoshift_classes_to_data_date(self, c: pd.DataFrame) -> pd.DataFrame:
        """
        If classes and data have disjoint dates (e.g., classes are 2025-09-26 but
        OD.xlsx parsed as today's date or only time-of-day), try shifting all class
        rows by the delta between their first date and the data's first date.
        Returns a possibly shifted copy of c.
        """
        if not len(self.ts_dt):
            return c

        data_start = min(self.ts_dt)
        data_end   = max(self.ts_dt)

        # If we already overlap, do nothing.
        c0, c1 = c["start"].min(), c["end"].max()
        if not (c1 < data_start or c0 > data_end):
            return c

        # Compute whole-day shift based on date (ignore time-of-day)
        try:
            delta_days = (data_start.normalize() - c0.normalize()).days
            if abs(delta_days) > 0:  # shift by full days
                c = c.copy()
                c["start"] = c["start"] + pd.Timedelta(days=delta_days)
                c["end"]   = c["end"]   + pd.Timedelta(days=delta_days)
                App.status(f"Class timestamps auto-shifted by {delta_days} day(s) to match data date.")
        except Exception:
            pass
        return c

    def load_data(self, path: str):
        """Load time-series data from CSV/XLSX with robust time/OD detection."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        self.path = path

        df = self._read_any_table(path)
        if df is None or df.empty:
            raise ValueError("Empty file.")

        cols = list(df.columns)
        if not cols:
            raise ValueError("No columns found.")

        # --- Pick time column (prefer known names, e.g., t_stamp) ---
        tcol = pick(cols, DATA_COL_GUESSES["time"]) or cols[0]

        # --- Pick OD column ---
        # 1) Try your previously preferred exact header (if you set one)
        EXACT_OD_COLUMN = globals().get("EXACT_OD_COLUMN", None)
        ENFORCE_EXACT_OD = globals().get("ENFORCE_EXACT_OD", False)

        ycol = None
        if EXACT_OD_COLUMN and EXACT_OD_COLUMN in df.columns:
            ycol = EXACT_OD_COLUMN
        else:
            # 2) Try alias list (includes Tag_value)
            ycol = pick(cols, DATA_COL_GUESSES["od"])

            # 3) If still unknown: choose the first numeric column that isn't the time column
            if ycol is None:
                for c in cols:
                    if c != tcol and pd.api.types.is_numeric_dtype(df[c]):
                        ycol = c
                        break

            # 4) If still unknown and there are only 2 columns, pick the other one
            if ycol is None and len(cols) == 2:
                ycol = cols[0] if cols[1] == tcol else cols[1]

            # If user insisted on exact name and we failed, raise a friendly error
            if ENFORCE_EXACT_OD and EXACT_OD_COLUMN and ycol != EXACT_OD_COLUMN:
                raise ValueError(
                    f"Could not find OD column '{EXACT_OD_COLUMN}' in this file.\n"
                    f"Columns present: {list(df.columns)}\n"
                    f"Tip: set ENFORCE_EXACT_OD=False to let me auto-detect (will use '{ycol}')."
                )

        if ycol is None:
            raise ValueError(
                "Could not determine OD column.\n"
                f"Columns present: {list(df.columns)}\n"
                "Try renaming your OD column to one of: "
                f"{DATA_COL_GUESSES['od']}"
            )

        # --- Coerce OD to numeric and drop NaNs; align TS accordingly ---
        od_series = self._smart_to_numeric(df[ycol])
        keep_mask = od_series.notna()
        if keep_mask.sum() == 0:
            raise ValueError(
                f"OD column '{ycol}' parsed to all NaN. "
                "Likely due to unexpected formatting. "
                "Try opening the file and saving as CSV, or send a sample of the OD cells."
            )

        # Optional: if all zeros, warn loudly (this is what led to the flat line)
        if (od_series[keep_mask] == 0).all():
            raise ValueError(
                f"OD column '{ycol}' parsed but all values are 0. "
                "This usually means decimal commas/units weren’t handled—"
                "please verify a few raw cell values.")

        self.od = od_series[keep_mask].tolist()
        self.ts = df.loc[keep_mask, tcol].astype(str).tolist()

        try:
            v = od_series[keep_mask]
            App.status(f"Using time='{tcol}', OD='{ycol}' • rows={len(v)} "
                    f"• min={v.min():.6g}, max={v.max():.6g}, mean={v.mean():.6g}")
        except Exception:
            pass


        # timestamps (as strings for UI) for the same rows
        self.ts = df.loc[keep_mask, tcol].astype(str).tolist()

        # Parsed timestamps (for class index mapping); synthesize if mostly bad
        try:
            ts_dt = pd.to_datetime(df.loc[keep_mask, tcol], errors="coerce")
        except Exception:
            ts_dt = pd.Series([pd.NaT] * keep_mask.sum())

        if ts_dt.isna().sum() > 0.9 * len(ts_dt):
            base = pd.Timestamp.utcnow()
            self.ts_dt = [base + pd.Timedelta(seconds=i) for i in range(len(self.od))]
        else:
            self.ts_dt = ts_dt.tolist()

        self.last_loaded_rows = len(self.od)
        self.path = path

        # Optional: let the status bar show what we picked
        try:
            App.status(f"Using time='{tcol}', OD='{ycol}'. Rows kept: {self.last_loaded_rows}.")
        except Exception:
            pass

        self.auto_classify(fs=1.0, win_sec=60, step_sec=30)

            # Map labels -> an NG risk in [0..1] (tune if you like)
    CLASS_TO_RISK = {
        "STEADY":        0.05,
        "MILD_WAVE":     0.40,
        "STRONG_WAVE":   0.90,
        "DRIFT":         0.80,
        "BURSTY_NOISY":  0.85,
        "UNCERTAIN":     0.50,
    }

    def current_timestamp(self):
        """Timestamp of the latest OD sample (pd.Timestamp or None)."""
        if not self.ts_dt:
            return None
        return self.ts_dt[-1]

    def current_class(self):
        """
        Return (label, risk_float) for the current timestamp, if it falls
        inside a loaded class segment; else (None, None).
        """
        if not self.classes or not self.ts_dt:
            return None, None
        t = self.ts_dt[-1]
        # Fast path using precomputed i0..i1 spans
        i_last = len(self.od) - 1
        for seg in reversed(self.classes):
            if seg["i0"] <= i_last < seg["i1"]:
                lbl = seg["label"]
                return lbl, self.CLASS_TO_RISK.get(lbl, 0.5)
        return None, None

    def load_classes_from_features(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if not self.od or not self.ts_dt:
            raise ValueError("Load OD data first.")
        ext = os.path.splitext(path.lower())[1]
        f = pd.read_excel(path, sheet_name=0) if ext in [".xlsx", ".xls"] else pd.read_csv(path)
        if f is None or f.empty:
            self.classes = []; App.status("Features file is empty."); return

        colmap = {str(k).strip().lower(): k for k in f.columns}
        def pick(*names):
            for n in names:
                if n in colmap: return colmap[n]
            return None

        start_col = pick("start","window_start","timestamp","time_start")
        end_col   = pick("end","window_end","time_end","stop")
        label_col = pick("label","class","wave_class","label_name","class_name","state","segment","category")
        if not start_col: raise ValueError("Features missing a 'start' column.")
        if not end_col:   raise ValueError("Features missing an 'end' column.")

        f["start"] = pd.to_datetime(f[start_col], errors="coerce")
        f["end"]   = pd.to_datetime(f[end_col],   errors="coerce")
        if label_col:
            f["label"] = f[label_col].astype(str).map(normalize_label)
        else:
            f["label"] = "UNCERTAIN"

        # clip to data span
        f = f.dropna(subset=["start","end"]); f = f[f["end"] > f["start"]].sort_values(["start","end"])
        smin, smax = min(self.ts_dt), max(self.ts_dt)
        f["start"] = f["start"].clip(lower=smin, upper=smax)
        f["end"]   = f["end"].clip(lower=smin, upper=smax)
        f = f[f["end"] > f["start"]]

                # --- normalize / infer labels ----------------------
        if label_col:
            src = f[label_col]
            # If the column is numeric IDs, map them to names
            if pd.api.types.is_numeric_dtype(src):
                id_map = {0:"STEADY", 1:"MILD_WAVE", 2:"STRONG_WAVE", 3:"DRIFT", 4:"BURSTY_NOISY", 5:"UNCERTAIN"}
                f["label"] = src.map(lambda v: id_map.get(int(v), "UNCERTAIN"))
            else:
                f["label"] = src.astype(str).map(normalize_label)
        else:
            # Try one-hot columns like STEADY/MILD_WAVE/... if present
            onehot_cols = [k for k in f.columns if str(k).strip().upper().replace(" ","_") in CLASS_COLORS.keys()]
            if onehot_cols:
                def pick_onehot(row):
                    for k in onehot_cols:
                        val = row[k]
                        try:
                            on = float(val) > 0.5
                        except Exception:
                            on = str(val).strip() in ("1","true","True","YES")
                        if on:
                            return normalize_label(str(k))
                    return "UNCERTAIN"
                f["label"] = f.apply(pick_onehot, axis=1)
            else:
                # last resort
                f["label"] = "UNCERTAIN"


        # precompute spans
        ns = np.array([int(t.value) for t in self.ts_dt], dtype=np.int64)
        if not (np.all(np.diff(ns) >= 0)):
            order = np.argsort(ns); ns = ns[order]
            self.ts_dt = [self.ts_dt[i] for i in order]; self.od = [self.od[i] for i in order]

        spans = []
        for _, row in f.iterrows():
            lbl = row["label"] if row["label"] in CLASS_COLORS else "UNCERTAIN"
            i0 = int(np.searchsorted(ns, int(row["start"].value), side="left"))
            i1 = int(np.searchsorted(ns, int(row["end"].value),   side="right"))
            i0 = max(0, min(i0, len(ns)-1)); i1 = max(0, min(i1, len(ns)))
            if i1 > i0: spans.append({"start": row["start"], "end": row["end"], "label": lbl, "i0": i0, "i1": i1})
        self.classes = spans
        App.status(f"Classes from features loaded: {len(self.classes)} spans.")

    def auto_classify(self, fs=1.0, win_sec=60, step_sec=30):
        """
        Heuristic 5-class segmentation from the loaded OD signal.
        Fills self.classes with spans having labels in:
        STEADY, MILD_WAVE, STRONG_WAVE, DRIFT, BURSTY_NOISY
        """
        if not self.od:
            self.classes = []
            return

        x = np.asarray(self.od, dtype=float)
        n = len(x)

        # --- smooth baseline (robust moving average) ---
        win = max(8, int(win_sec * fs))
        step = max(1, int(step_sec * fs))
        if win >= n:
            win = max(8, n // 10)
        baseline = pd.Series(x).rolling(win, min_periods=max(3, win//5)).mean().to_numpy()
        baseline = np.nan_to_num(baseline, nan=np.nanmedian(x))
        resid = x - baseline

        # precompute window indices
        starts = list(range(0, n - win + 1, step))
        if not starts:
            starts = [0]
        stops = [min(s + win, n) for s in starts]

        # features per window
        feats = []
        for s, e in zip(starts, stops):
            y = x[s:e]
            r = resid[s:e]

            # slope on raw (trend)
            k = len(y)
            sx = k*(k-1)/2.0
            sxx = k*(k-1)*(2*k-1)/6.0
            sy = float(np.sum(y))
            sxy = float(np.sum(np.arange(k)*y))
            denom = k*sxx - sx*sx
            slope = 0.0 if abs(denom) < 1e-12 else (k*sxy - sx*sy)/denom

            r_ptp = float(np.max(r) - np.min(r))
            r_mad = float(np.mean(np.abs(r - np.median(r))))
            r_std = float(np.std(r))
            # high-freq “noisiness” proxy
            diff_std = float(np.std(np.diff(y))) if k > 2 else 0.0

            feats.append((s, e, slope, r_ptp, r_mad, r_std, diff_std))

        # robust thresholds from medians
        abs_slopes = np.array([abs(f[2]) for f in feats])
        r_ptps    = np.array([f[3] for f in feats])
        r_mads    = np.array([f[4] for f in feats])
        r_stds    = np.array([f[5] for f in feats])
        dstds     = np.array([f[6] for f in feats])

        eps = 1e-12
        slope_thr = np.median(abs_slopes) + 3.0*np.median(np.abs(abs_slopes - np.median(abs_slopes)))
        noise_thr = np.median(r_stds) + 2.5*np.median(np.abs(r_stds - np.median(r_stds)))
        # wave thresholds based on residual amplitude
        mild_thr   = np.median(r_ptps) + 1.5*np.median(np.abs(r_ptps - np.median(r_ptps)))
        strong_thr = np.median(r_ptps) + 3.0*np.median(np.abs(r_ptps - np.median(r_ptps)))

        spans = []
        for (s, e, slope, r_ptp, r_mad, r_std, diff_std) in feats:
            # decision order matters
            if abs(slope) > max(slope_thr, 1e-9) and r_ptp < strong_thr:
                lbl = "DRIFT"
            elif r_std > noise_thr and r_ptp < strong_thr:
                lbl = "BURSTY_NOISY"
            elif r_ptp >= strong_thr:
                lbl = "STRONG_WAVE"
            elif r_ptp >= mild_thr:
                lbl = "MILD_WAVE"
            else:
                lbl = "STEADY"

            spans.append({"i0": int(s), "i1": int(e), "label": lbl,
                          "start": self.ts_dt[s] if self.ts_dt else pd.NaT,
                          "end":   self.ts_dt[e-1] if self.ts_dt else pd.NaT})

        # Merge adjacent windows with the same label to make cleaner bands
        merged = []
        for seg in spans:
            if not merged or seg["label"] != merged[-1]["label"] or seg["i0"] > merged[-1]["i1"]:
                merged.append(seg)
            else:
                merged[-1]["i1"] = seg["i1"]
                merged[-1]["end"] = seg["end"]

        self.classes = merged
        App.status(f"Auto-classes computed: {len(self.classes)} span(s).")


    # ---- math helpers (no numpy) ----
    @staticmethod
    def _linreg_slope(y):
        """Least-squares slope over y with x = 0..n-1 (pure Python)."""
        n = len(y)
        if n < 2: return 0.0
        sx = n*(n-1)/2.0
        sxx = n*(n-1)*(2*n-1)/6.0
        sy = sum(y)
        sxy = sum(i*yi for i, yi in enumerate(y))
        denom = n*sxx - sx*sx
        if abs(denom) < 1e-12: return 0.0
        return (n*sxy - sx*sy) / denom

    def recent_window(self, n=1024):
        if not self.od: return []
        n = min(n, len(self.od))
        return self.od[-n:]

    def trend_slope(self, n=1024):
        y = self.recent_window(n)
        return self._linreg_slope(y) if y else 0.0

    def volatility_p2p(self, n=1024):
        y = self.recent_window(n)
        return (max(y) - min(y)) if y else 0.0

    def ng_score(self, n=1024, spec_mm=10.0, spec_band=0.02):
        """
        Simple placeholder score in [0..100]:
          - normalize slope by spec band
          - normalize peak-to-peak by spec band
        Combine as weighted sum (tweak weights as needed).
        """
        slope = abs(self.trend_slope(n))
        p2p   = self.volatility_p2p(n)
        slope_norm = min(1.0, slope / (spec_band/200.0 + 1e-9))
        p2p_norm   = min(1.0, p2p   / (spec_band*2.0 + 1e-9))
        score = 100.0 * (0.6 * p2p_norm + 0.4 * slope_norm)
        return max(0.0, min(100.0, score))

    # ----- class windows support -----
    def _ts_values_ns(self):
        """Convert timestamps to int64 ns since epoch for fast searchsorted."""
        out = []
        for t in self.ts_dt:
            if isinstance(t, pd.Timestamp) and pd.notna(t):
                out.append(int(t.value))
            else:
                out.append(np.iinfo(np.int64).min)
        return np.array(out, dtype=np.int64)

    def _time_to_index(self, t: pd.Timestamp):
        if not isinstance(t, pd.Timestamp) or not len(self.ts_dt):
            return 0
        ns = self._ts_values_ns()
        key = int(t.value)
        i = np.searchsorted(ns, key, side="left")
        if i <= 0: return 0
        if i >= len(ns): return len(ns)-1
        return i if abs(ns[i]-key) < abs(ns[i-1]-key) else (i-1)

    def load_classes(self, path: str):
        """
        Load waviness class segments from CSV/XLS/XLSX.
        Expected flexible headers:
        start: start | window_start | timestamp | time_start
        end:   end | window_end | time_end | stop
        label: label | class | wave_class
        Produces self.classes = list of dicts with sample spans {start,end,label,i0,i1}.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if not self.od or not self.ts_dt:
            raise ValueError("Load OD data first (so we can align class windows to samples).")

        # read table
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            c = pd.read_excel(path, sheet_name=0)
        else:
            c = pd.read_csv(path)
        if c is None or c.empty:
            self.classes = []
            App.status("Classes file is empty.")
            return

        # helper: column picker on lowercase names
        colmap = {str(k).strip().lower(): k for k in c.columns}
        def pick(*names):
            for n in names:
                if n in colmap:
                    return colmap[n]
            return None

        start_col = pick("start","window_start","timestamp","time_start")
        end_col   = pick("end","window_end","time_end","stop")
        label_col = pick("label","class","wave_class","label_name","class_name","state","segment","category")
        if not start_col:
            raise ValueError("Classes file missing a 'start' (or equivalent) column.")

        c["start"] = pd.to_datetime(c[start_col], errors="coerce")
        if end_col:
            c["end"] = pd.to_datetime(c[end_col], errors="coerce")
        else:
            c = c.sort_values("start")
            gaps = c["start"].diff().dropna()
            win = gaps.median() if len(gaps) else pd.Timedelta(minutes=5)
            c["end"] = c["start"].shift(-1)
            c.loc[c["end"].isna(), "end"] = c["start"] + win
        # --- normalize / infer labels ----------------------
        if label_col:
            src = c[label_col]
            # If the column is numeric IDs, map them to names
            if pd.api.types.is_numeric_dtype(src):
                id_map = {0:"STEADY", 1:"MILD_WAVE", 2:"STRONG_WAVE", 3:"DRIFT", 4:"BURSTY_NOISY", 5:"UNCERTAIN"}
                c["label"] = src.map(lambda v: id_map.get(int(v), "UNCERTAIN"))
            else:
                c["label"] = src.astype(str).map(normalize_label)
        else:
            # Try one-hot columns like STEADY/MILD_WAVE/... if present
            onehot_cols = [k for k in c.columns if str(k).strip().upper().replace(" ","_") in CLASS_COLORS.keys()]
            if onehot_cols:
                def pick_onehot(row):
                    for k in onehot_cols:
                        val = row[k]
                        try:
                            on = float(val) > 0.5
                        except Exception:
                            on = str(val).strip() in ("1","true","True","YES")
                        if on:
                            return normalize_label(str(k))
                    return "UNCERTAIN"
                c["label"] = c.apply(pick_onehot, axis=1)
            else:
                # last resort
                c["label"] = "UNCERTAIN"

        # normalize tz so we can compare with self.ts_dt (which are tz-naive)
        for col in ["start","end"]:
            if hasattr(c[col], "dt"):
                try:    c[col] = c[col].dt.tz_convert(None)
                except: c[col] = c[col].dt.tz_localize(None)

        # labels
        if label_col:
            c["label"] = c[label_col].astype(str).map(normalize_label)
        else:
            c["label"] = "UNCERTAIN"

        # clean & sort
        c = c.dropna(subset=["start","end"])
        c = c[c["end"] > c["start"]].sort_values(["start","end"]).reset_index(drop=True)

        # --- AUTO-SHIFT if classes & data don't overlap by date ---
        c = self._try_autoshift_classes_to_data_date(c)

        # clip to data span
        smin, smax = min(self.ts_dt), max(self.ts_dt)
        c["start"] = c["start"].clip(lower=smin, upper=smax)
        c["end"]   = c["end"].clip(lower=smin, upper=smax)
        c = c[c["end"] > c["start"]]

        # recompute spans on the (maybe shifted) table
        ns = np.array([int(t.value) for t in self.ts_dt], dtype=np.int64)
        if not (np.all(np.diff(ns) >= 0)):
            order = np.argsort(ns)
            ns = ns[order]
            self.ts_dt = [self.ts_dt[i] for i in order]
            self.od    = [self.od[i]    for i in order]

        spans = []
        for _, row in c.iterrows():
            lbl = str(row["label"])
            if lbl not in CLASS_COLORS: lbl = "UNCERTAIN"
            s_ns = int(row["start"].value); e_ns = int(row["end"].value)
            i0 = int(np.searchsorted(ns, s_ns, side="left"))
            i1 = int(np.searchsorted(ns, e_ns, side="right"))
            i0 = max(0, min(i0, len(ns) - 1))
            i1 = max(0, min(i1, len(ns)))
            if i1 <= i0: continue
            spans.append({"start": row["start"], "end": row["end"], "label": lbl, "i0": i0, "i1": i1})

        self.classes = spans
        App.status(f"Classes loaded: {len(self.classes)} span(s).")


DATA = DataStore()
# ========================= Widgets =========================
class Gauge(ttk.Frame):
    def __init__(self, parent, width=360, height=200, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.width, self.height = width, height
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        self._needle = None
        self._pct_label = ttk.Label(self, text="— %", font=("Segoe UI", 12, "bold"))
        self._pct_label.grid(row=1, column=0, pady=(8,0))
        self._status = ttk.Label(self, text="Status: —", font=("Segoe UI", 11))
        self._status.grid(row=2, column=0)
        self._draw_static()
        self.set_value(0)

    def _draw_static(self):
        w, h = self.width, self.height
        cx, cy, r = w//2, h-10, min(w, h*2)//2 - 10
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=60, fill="#16A34A", outline="")
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=240, extent=60, fill="#D97706", outline="")
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=300, extent=60, fill="#DC2626", outline="")
        for i in range(0, 11):
            ang = math.radians(180 + i*18)
            x0 = cx + (r-18)*math.cos(ang); y0 = cy + (r-18)*math.sin(ang)
            x1 = cx + (r-2)*math.cos(ang);  y1 = cy + (r-2)*math.sin(ang)
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill="#334155")
        self.canvas.create_text(cx - r + 40, cy - 20, text="OK", font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(cx + r - 40, cy - 20, text="NG", font=("Segoe UI", 11, "bold"))
        self._cx, self._cy, self._r = cx, cy, r

    def set_value(self, pct: float):
        pct = max(0.0, min(100.0, float(pct)))
        ang = math.radians(180 + 180 * pct / 100.0)
        cx, cy, r = self._cx, self._cy, self._r
        x = cx + (r-26) * math.cos(ang)
        y = cy + (r-26) * math.sin(ang)
        if self._needle: self.canvas.delete(self._needle)
        self._needle = self.canvas.create_line(cx, cy, x, y, width=5, fill="#111827", capstyle=tk.ROUND)
        self._pct_label.config(text=f"{pct:0.1f}% confidence")
        status = "NG" if pct >= 50 else "OK"
        self._status.config(text=f"Status: {status}",
                            foreground=("#DC2626" if status=="NG" else "#16A34A"))

class TrendChart(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.bind("<Configure>", lambda e: self.redraw())

    def redraw(self):
        self.canvas.delete("all")
        w = max(200, self.canvas.winfo_width())
        h = max(140, self.canvas.winfo_height())
        pad = 16
        self.canvas.create_rectangle(0,0,w,h, fill="white", outline="")
        self.canvas.create_rectangle(pad,pad,w-pad,h-pad, outline="#CBD5E1")

        y = DATA.recent_window(1200)
        if len(y) < 5:
            self.canvas.create_text(w//2, h//2, text="(Load CSV to see history)", fill="#6B7280")
            return

        ymin, ymax = min(y), max(y)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0

        def X(i):  return pad + (w-2*pad) * (i/(len(y)-1))
        def Y(v):  return h-pad - (h-2*pad) * ((v - ymin)/(ymax - ymin))

        # raw line
        for i in range(1, len(y)):
            self.canvas.create_line(X(i-1), Y(y[i-1]), X(i), Y(y[i]), fill="#93C5FD", width=1)

        # smoothed line
        k = max(5, len(y)//50)
        sm, s = [], 0.0
        for i, v in enumerate(y):
            s += v
            if i >= k: s -= y[i-k]
            sm.append(s / min(i+1, k))
        for i in range(1, len(sm)):
            self.canvas.create_line(X(i-1), Y(sm[i-1]), X(i), Y(sm[i]), fill="#2563EB", width=2)

        # slope badge
        slope = DATA.trend_slope(min(1024, len(y)))
        color = "#DC2626" if slope > 0 else ("#16A34A" if slope < 0 else "#6B7280")
        label = "Uptrend" if slope > 0 else ("Downtrend" if slope < 0 else "Stable")
        self.canvas.create_text(w-pad-70, pad+14, text=label, fill=color, font=("Segoe UI", 10, "bold"))
        ax = w - pad - 30; ay = pad + 28
        dy = -16 if slope > 0 else (16 if slope < 0 else 0)
        self.canvas.create_line(ax-10, ay, ax+10, ay+dy, arrow=tk.LAST, width=3, fill=color)

        # --- CLASS OVERLAYS (stipple ~= alpha) ---
        # --- CLASS OVERLAYS (shaded bands using stipple ~= alpha) ---
        if getattr(DATA, "classes", None):
            y0 = pad + 1
            y1 = h - pad - 1
            n_total = len(DATA.od)
            offset = n_total - len(y)  # global index represented by local x==0
            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                # ----- index clamp that always yields width >= 1 -----
                i0 = seg["i0"] - offset
                i1 = seg["i1"] - offset
                if i1 <= 0 or i0 >= len(y):
                    continue
                i0 = max(0, min(i0, len(y) - 2))     # allow room for at least 1 sample
                i1 = max(i0 + 1, min(i1, len(y) - 1))

                x0 = X(i0); x1 = X(i1)
                color = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple="gray25")
                self.canvas.create_text(x0+4, y0+10, text=seg["label"], anchor="w",
                                        fill="#333333", font=("Segoe UI", 8, "bold"))



        # legend
        legend_x = pad + 6
        legend_y = pad + 10
        for idx, (name, col) in enumerate(CLASS_COLORS.items()):
            if name not in VISIBLE_CLASSES: continue
            self.canvas.create_rectangle(legend_x, legend_y + idx*16 - 6,
                                         legend_x+12, legend_y + idx*16 + 6,
                                         fill=col, width=0, stipple="gray25")
            self.canvas.create_text(legend_x+18, legend_y + idx*16, text=name, anchor="w",
                                    fill="#111827", font=("Segoe UI", 8))

class FeatureEngine:
    def __init__(self, od, window=30, fs=1.0):
        self.od = np.asarray(od, dtype=float)
        self.win = max(8, int(window * fs))
        self.step = self.win // 2  # 50% overlap
        self.fs = fs

    def _windows(self):
        x = self.od
        for i in range(0, len(x)-self.win+1, self.step):
            yield x[i:i+self.win]

    def feature_table(self):
        rows = []
        for w in self._windows():
            y = w
            mean = float(np.mean(y))
            std  = float(np.std(y))
            p2p  = float(np.max(y) - np.min(y))
            rel_range = p2p / max(1e-9, mean)
            mad = float(rolling_mad(y))
            rel_max = (np.max(y) - mean) / max(1e-9, mean)
            rel_min = (mean - np.min(y)) / max(1e-9, mean)
            coef_var = std / max(1e-9, mean)
            norm_var = (std**2) / max(1e-9, mean**2)
            ptp_ratio = p2p / max(1e-9, std)
            rows.append({
                "mean_abs_diff": mad,
                "coef_variation": coef_var,
                "relative_range": rel_range,
                "normalized_variance": norm_var,
                "peak_to_peak_ratio": ptp_ratio,
                "relative_max_deviation": rel_max,
                "relative_min_deviation": rel_min,
                "max_abs_diff": p2p,
            })
        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def demo_labels(self, df):
        thr = df["mean_abs_diff"].median() + df["mean_abs_diff"].std()
        return (df["mean_abs_diff"] > thr).astype(int)

    def kfold_curve(self, X, y, model, windows=[10,20,30,40,60,90,120]):
        sizes, accs = [], []
        for w in windows:
            self.win = max(8, int(w * self.fs))
            self.step = self.win // 2
            df = self.feature_table()
            if len(df) < 20: continue
            Xw = df.values
            yw = self.demo_labels(df) if y is None else y[:len(df)]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, te in skf.split(Xw, yw):
                Xtr, Xte = Xw[tr], Xw[te]
                ytr = yw.iloc[tr] if isinstance(yw, pd.Series) else yw[tr]
                yte = yw.iloc[te] if isinstance(yw, pd.Series) else yw[te]
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
                model.fit(Xtr, ytr)
                scores.append(model.score(Xte, yte))
            sizes.append(w); accs.append(np.mean(scores))
        return np.array(sizes), np.array(accs)

    def fig_sixpack(self):
        df = self.feature_table()
        if df.empty:
            fig = Figure(figsize=(12,6), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5,0.5,"Not enough data for analysis.\nLoad a longer CSV.", ha="center", va="center")
            ax.axis("off")
            return fig

        y = self.demo_labels(df)
        fig = Figure(figsize=(12,6), dpi=100)
        gs = fig.add_gridspec(2,3, wspace=0.35, hspace=0.35)

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        sizes, acc_rf = self.kfold_curve(df.values, y, clf)
        ax1 = fig.add_subplot(gs[0,0])
        for mult, lab in [(0.98,"Logistic Regression (demo)"), (1.00,"Random Forest"), (0.995,"SVM (demo)")]:
            ax1.plot(sizes, np.minimum(1.0, acc_rf*mult), marker="o", label=lab)
        ax1.set_title("Average model accuracy vs. Window size\n5-split K-Folds")
        ax1.set_xlabel("Window size (sec)"); ax1.set_ylabel("Average accuracy"); ax1.legend()

        scaler = StandardScaler().fit(df.values)
        Xs = scaler.transform(df.values)
        clf.fit(Xs, y)
        importances = clf.feature_importances_
        order = np.argsort(importances)
        ax2 = fig.add_subplot(gs[0,1])
        ax2.barh(np.array(df.columns)[order], importances[order])
        ax2.set_title("Random Forest feature importance\nWindow size: ~30 sec")
        ax2.set_xlabel("Feature weight")

        ax3 = fig.add_subplot(gs[0,2])
        ax3.hist(df.loc[y==0,"mean_abs_diff"], bins=40, alpha=0.6, label="Good")
        ax3.hist(df.loc[y==1,"mean_abs_diff"], bins=40, alpha=0.6, label="Rejected, chatter")
        ax3.set_title("Mean Absolute Difference (smoothness indicator)\nWindow size: ~30 sec")
        ax3.set_xlabel("Mean Absolute Difference"); ax3.set_ylabel("Density"); ax3.legend()

        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        ax4 = fig.add_subplot(gs[1,0])
        ax4.scatter(Xp[y==0,0], Xp[y==0,1], s=8, alpha=0.5, label="Good")
        ax4.scatter(Xp[y==1,0], Xp[y==1,1], s=8, alpha=0.8, label="Bad (Chatter)", color="red")
        ax4.set_title("Principal component analysis")
        ax4.set_xlabel(f"PC1: {pca.explained_variance_ratio_[0]*100:0.1f}% variance")
        ax4.set_ylabel(f"PC2: {pca.explained_variance_ratio_[1]*100:0.1f}% variance")
        ax4.legend(loc="best")

        ax5 = fig.add_subplot(gs[1,1])
        comps = pca.components_
        ax5.axvline(0,color="gray",linewidth=0.8); ax5.axhline(0,color="gray",linewidth=0.8)
        for i, name in enumerate(df.columns):
            ax5.arrow(0, 0, comps[0,i], comps[1,i], head_width=0.02, head_length=0.02, fc='k', ec='k', alpha=0.6)
            ax5.text(comps[0,i]*1.08, comps[1,i]*1.08, name, fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="#FFF9C4", ec="#999"))
        ax5.set_xlim(-1,1); ax5.set_ylim(-1,1)
        ax5.set_title("Feature weights on principal components")
        ax5.set_xlabel("PC1 weights"); ax5.set_ylabel("PC2 weights")

        ax6 = fig.add_subplot(gs[1,2])
        corr = df.corr().values
        im = ax6.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax6.set_xticks(range(len(df.columns))); ax6.set_xticklabels(df.columns, rotation=60, ha="right", fontsize=8)
        ax6.set_yticks(range(len(df.columns))); ax6.set_yticklabels(df.columns, fontsize=8)
        ax6.set_title("Feature Correlation Matrix")
        fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04).set_label("Correlation")
        return fig

# ========================= Pages =========================
class BasePage(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, padding=16, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(99, weight=1)
    def headline(self, text):
        lbl = ttk.Label(self, text=text, style="Headline.TLabel")
        lbl.grid(row=0, column=0, sticky="w", pady=(0, 12))
        return lbl
    def placeholder(self, parent, text):
        box = ttk.Label(parent, text=text, style="Placeholder.TLabel",
                        anchor="center", padding=24, relief="ridge")
        box.grid(sticky="nsew")
        return box

class DataPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Data")
        controls = ttk.Frame(self); controls.grid(row=1, column=0, sticky="ew", pady=(0,12))
        for i in range(12): controls.columnconfigure(i, weight=1)
        # DataPage.__init__ controls block

        ttk.Button(controls, text="Load CSV/XLSX…",        command=self.load_csv         ).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Load Classes…",         command=self.load_classes     ).grid(row=0, column=1, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Load from Features…",   command=self.load_from_features).grid(row=0, column=2, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Run Preprocess",        command=lambda: App.busy("Preprocess (todo)")).grid(row=0, column=3, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Compute FFT",           command=lambda: App.busy("FFT (todo)")        ).grid(row=0, column=4, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Open in VS Code",       command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=5, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Class Diagnostics",     command=self.show_diag        ).grid(row=0, column=6, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Auto Classify",         command=lambda: (DATA.auto_classify(), App.status("Auto classification done.")))\
            .grid(row=0, column=7, sticky="w", padx=(0,8))

        self.info = ttk.Label(self, text="No file loaded.", foreground="#6B7280")
        self.info.grid(row=2, column=0, sticky="w")

        table_area = ttk.Frame(self); table_area.grid(row=3, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(3, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select data file (CSV/XLSX/XLS)",
            filetypes=[("Data files", "*.csv *.xlsx *.xls"),
                       ("CSV files", "*.csv"),
                       ("Excel files", "*.xlsx *.xls"),
                       ("All files","*.*")]
        )
        if not path: return
        try:
            DATA.load_data(path)
            self.info.config(text=f"Loaded: {os.path.basename(path)}  •  rows={len(DATA.od)}")
            App.status("Data loaded. History & gauge now using real data.")
        except Exception as e:
            messagebox.showerror("Load Data failed", str(e))
            App.status("Data load failed")

    def load_classes(self):
        if not DATA.od:
            messagebox.showwarning("Load data first", "Please load a data file before loading classes.")
            return
        path = filedialog.askopenfilename(
            title="Select classes file (CSV/XLSX/XLS)",
            filetypes=[("Data files", "*.csv *.xlsx *.xls"),
                       ("CSV files", "*.csv"),
                       ("Excel files", "*.xlsx *.xls"),
                       ("All files","*.*")]
        )
        if not path: return
        try:
            DATA.load_classes(path)
            self.info.config(text=f"{self.info.cget('text')}  •  classes={len(DATA.classes)}")
            App.status(f"Classes loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Classes failed", str(e))
            App.status("Classes load failed")

    def show_diag(self):
        if not DATA.od:
            messagebox.showinfo("Diagnostics", "Load a data file first."); return
        data_min = min(DATA.ts_dt) if DATA.ts_dt else None
        data_max = max(DATA.ts_dt) if DATA.ts_dt else None
        n_spans = len(getattr(DATA, "classes", []))
        cls_min = min([s["start"] for s in DATA.classes]) if n_spans else None
        cls_max = max([s["end"]   for s in DATA.classes]) if n_spans else None
        msg = [
            f"Data rows: {len(DATA.od)}",
            f"Data time range: {data_min} → {data_max}",
            f"Class spans: {n_spans}",
            f"Class time range: {cls_min} → {cls_max}",
        ]
        messagebox.showinfo("Diagnostics", "\n".join(msg))

    def load_from_features(self):
        if not DATA.od:
            messagebox.showwarning("Load data first", "Please load a data file before features."); return
        path = filedialog.askopenfilename(
            title="Select features file (CSV/XLSX/XLS)",
            filetypes=[("Data files","*.csv *.xlsx *.xls"),("CSV files","*.csv"),
                    ("Excel files","*.xlsx *.xls"),("All files","*.*")]
        )
        if not path: return
        try:
            DATA.load_classes_from_features(path)
            self.info.config(text=f"{self.info.cget('text')}  •  classes={len(DATA.classes)}")
            App.status(f"Classes (from features) loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load from Features failed", str(e)); App.status("Load from features failed")



class LiveTimeSeries(ttk.Frame):
    """Matplotlib live plot with class shading via axvspan."""
    def __init__(self, parent):
        super().__init__(parent)
        self.fig = Figure(figsize=(6,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._last_len = -1
        self.after(1000, self._tick)

    def _draw(self):
        self.ax.clear()
        if not DATA.od:
            self.ax.text(0.5,0.5,"(Load CSV to see live plot)", ha="center", va="center")
            self.ax.axis("off")
            self.canvas.draw(); return

        y = np.asarray(DATA.od[-2400:])  # last ~2400 samples
        x = np.arange(len(y))
        self.ax.plot(x, y, linewidth=1.0, alpha=0.7, label="OD")

        # simple smoothing
        k = max(5, len(y)//50)
        if len(y) >= k:
            sm = pd.Series(y).rolling(k, min_periods=1).mean().values
            self.ax.plot(x, sm, linewidth=2.0, label="smooth")

        # class shading (convert global i0/i1 to local indices)
        # class shading (convert global i0/i1 to local indices)
        # class shading (convert global i0/i1 to local indices)
        if getattr(DATA, "classes", None):
            n_total = len(DATA.od)
            offset = n_total - len(y)
            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                i0 = seg["i0"] - offset
                i1 = seg["i1"] - offset
                if i1 <= 0 or i0 >= len(y):
                    continue
                # clamp so we always draw something
                i0 = max(0, min(i0, len(y) - 2))
                i1 = max(i0 + 1, min(i1, len(y) - 1))

                c = CLASS_COLORS.get(seg["label"], "#BBBBBB")  # <-- this was missing
                self.ax.axvspan(i0, i1, facecolor=c, alpha=0.25, linewidth=0)


        self.ax.set_title("OD vs Samples — live")
        self.ax.set_xlabel("sample index"); self.ax.set_ylabel("OD (mm)")
        self.ax.legend(loc="upper left", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def _tick(self):
        if len(DATA.od) != self._last_len:
            self._draw()
            self._last_len = len(DATA.od)
        self.after(1000, self._tick)

class ResultsPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Results (Blueprint)")

        tools = ttk.Frame(self); tools.grid(row=1, column=0, sticky="ew", pady=(0,12))
        ttk.Button(tools, text="Predict Latest", command=self.predict_latest).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Button(tools, text="Export Report", command=lambda: App.busy("Export report (todo)")).grid(row=0, column=1, sticky="w", padx=(0,8))
        ttk.Button(tools, text="Open in VS Code", command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=2, sticky="w")

        grid = ttk.Frame(self); grid.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        left = ttk.Frame(grid); left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,8))
        ttk.Label(left, text="OD vs Time (Live)", style="Subhead.TLabel").pack(anchor="w")
        self.live = LiveTimeSeries(left)
        self.live.pack(fill="both", expand=True, pady=(6,0))

        right_top = ttk.Frame(grid); right_top.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_top, text="OK / NG Indicator", style="Subhead.TLabel").pack(anchor="w")
        self.gauge = Gauge(right_top, width=360, height=200); self.gauge.pack(fill="both", expand=True, pady=(6,0))

        right_bot = ttk.Frame(grid); right_bot.grid(row=1, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_bot, text="Model Metrics", style="Subhead.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        right_bot.columnconfigure((0,1,2,3), weight=1); right_bot.rowconfigure(1, weight=1)

        mat = ttk.Frame(right_bot, padding=6, relief="ridge"); mat.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0,6))
        for i in range(3): mat.columnconfigure(i, weight=1)
        for i in range(3): mat.rowconfigure(i, weight=1)
        ttk.Label(mat, text="Predicted", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, columnspan=2)
        ttk.Label(mat, text="Actual", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, rowspan=2, sticky="s")
        ttk.Label(mat, text="OK").grid(row=1, column=1);   ttk.Label(mat, text="NG").grid(row=1, column=2)
        ttk.Label(mat, text="OK").grid(row=2, column=0, sticky="e")
        self.cm_tn = ttk.Label(mat, text="—", style="KPI.TLabel"); self.cm_tn.grid(row=2, column=1)
        self.cm_fp = ttk.Label(mat, text="—", style="KPI.TLabel"); self.cm_fp.grid(row=2, column=2)
        ttk.Label(mat, text="NG").grid(row=3, column=0, sticky="e")
        self.cm_fn = ttk.Label(mat, text="—", style="KPI.TLabel"); self.cm_fn.grid(row=3, column=1)
        self.cm_tp = ttk.Label(mat, text="—", style="KPI.TLabel"); self.cm_tp.grid(row=3, column=2)

        kpis = ttk.Frame(right_bot); kpis.grid(row=1, column=2, columnspan=2, sticky="nsew")
        for i in range(2): kpis.columnconfigure(i, weight=1)
        self.k_acc = ttk.Label(kpis, text="Accuracy: —", style="KPI.TLabel"); self.k_acc.grid(row=0, column=0, sticky="w", pady=(0,6))
        self.k_prec= ttk.Label(kpis, text="Precision: —", style="KPI.TLabel"); self.k_prec.grid(row=1, column=0, sticky="w", pady=(0,6))
        self.k_rec = ttk.Label(kpis, text="Recall: —",    style="KPI.TLabel"); self.k_rec.grid(row=0, column=1, sticky="w", pady=(0,6))
        self.k_f1  = ttk.Label(kpis, text="F1 Score: —",  style="KPI.TLabel"); self.k_f1.grid(row=1, column=1, sticky="w", pady=(0,6))

        self.pred_label = ttk.Label(right_top, text="Predicted class: —", style="KPI.TLabel")
        self.pred_label.pack(anchor="w", pady=(6, 0))
        self.pred_conf  = ttk.Label(right_top, text="Confidence: —", style="KPI.TLabel")
        self.pred_conf.pack(anchor="w")

        self.after(1000, self._tick)

    def predict_latest(self):
        lbl, risk = DATA.current_class()
        if lbl is None:
            risk = DATA.ng_score(n=1024, spec_mm=10.0, spec_band=0.02) / 100.0
            if   risk < 0.25: lbl = "STEADY"
            elif risk < 0.45: lbl = "MILD_WAVE"
            elif risk < 0.65: lbl = "DRIFT"
            elif risk < 0.80: lbl = "BURSTY_NOISY"
            else:             lbl = "STRONG_WAVE"

        self.pred_label.config(text=f"Predicted class: {lbl}")
        self.pred_conf.config(text=f"Confidence: {risk*100:0.1f}%")
    # (demo confusion/kpis stays the same)


        # If no class window overlaps the latest point, derive a risk from NG score.
        if lbl is None:
            risk = DATA.ng_score(n=1024, spec_mm=10.0, spec_band=0.02) / 100.0
            # map continuous risk to a label-ish category
            if   risk < 0.25: lbl = "STEADY"
            elif risk < 0.45: lbl = "MILD_WAVE"
            elif risk < 0.65: lbl = "DRIFT"
            elif risk < 0.80: lbl = "BURSTY_NOISY"
            else:             lbl = "STRONG_WAVE"

        # Show to the user
        self.pred_label.config(text=f"Predicted class: {lbl}")
        self.pred_conf.config(text=f"Confidence: {risk*100:0.1f}%")

        # Demo confusion-block numbers (keep your existing block if you want)
        tp = int(100 * risk * 0.6); tn = int(100 * (1.0 - risk) * 0.6)
        fp = int(100 * (1.0 - risk) * 0.4); fn = int(100 * risk * 0.4)
        acc = (tp + tn) / max(1, (tp+tn+fp+fn))
        prec = tp / max(1, (tp+fp)); rec = tp / max(1, (tp+fn))
        f1 = 2*prec*rec / max(1e-9, (prec+rec))
        self.cm_tp.config(text=str(tp)); self.cm_tn.config(text=str(tn))
        self.cm_fp.config(text=str(fp)); self.cm_fn.config(text=str(fn))
        self.k_acc.config(text=f"Accuracy: {acc*100:0.1f}%")
        self.k_prec.config(text=f"Precision: {prec*100:0.1f}%")
        self.k_rec.config(text=f"Recall: {rec*100:0.1f}%")
        self.k_f1.config(text=f"F1 Score: {f1*100:0.1f}%")
        App.status("Prediction updated from waviness class / risk.")


    def _tick(self):
        if DATA.od:
            lbl, class_risk = DATA.current_class()
            if class_risk is not None:
                pct = class_risk * 100.0
                self.pred_label.config(text=f"Predicted class: {lbl}")
                self.pred_conf.config(text=f"Confidence: {pct:0.1f}%")
            else:
                pct = DATA.ng_score(n=1024, spec_mm=10.0, spec_band=0.02)
        else:
            pct = 50 + 15 * math.sin(datetime.now().timestamp()/2.0)

        self.gauge.set_value(pct)
        self.after(1000, self._tick)



class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Trend")
        desc = "Trend view shows last N samples with smoothed curve and class overlays.\n" \
               "Use this to see if the process is drifting toward NG before defects happen."
        ttk.Label(self, text=desc, foreground="#6B7280").grid(row=1, column=0, sticky="w", pady=(0,8))

        grid = ttk.Frame(self); grid.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        grid.columnconfigure(0, weight=3); grid.columnconfigure(1, weight=2)
        grid.rowconfigure(0, weight=1)

        self.chart = TrendChart(grid); self.chart.grid(row=0, column=0, sticky="nsew", padx=(0,8))

        side = ttk.Frame(grid); side.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        side.columnconfigure(0, weight=1)
        ttk.Label(side, text="Latest Stats", style="Subhead.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        self.l_slope = ttk.Label(side, text="Slope: —", style="KPI.TLabel"); self.l_slope.grid(row=1, column=0, sticky="w", pady=4)
        self.l_p2p   = ttk.Label(side, text="Peak-to-peak: —", style="KPI.TLabel"); self.l_p2p.grid(row=2, column=0, sticky="w", pady=4)
        self.l_score = ttk.Label(side, text="Risk score: —", style="KPI.TLabel"); self.l_score.grid(row=3, column=0, sticky="w", pady=4)

        self.after(1000, self._tick)

    def _tick(self):
        slope = DATA.trend_slope(1024)
        p2p   = DATA.volatility_p2p(1024)
        score = DATA.ng_score(1024, spec_mm=10.0, spec_band=0.02)
        self.l_slope.config(text=f"Slope: {slope:0.6f} mm/sample")
        self.l_p2p.config(text=f"Peak-to-peak: {p2p:0.4f} mm")
        self.l_score.config(text=f"Risk score: {score:0.1f}/100")
        self.chart.redraw()
        self.after(1000, self._tick)

class TestsPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Tests")
        row = ttk.Frame(self); row.grid(row=1, column=0, sticky="w", pady=(0,12))
        ttk.Button(row, text="Test Button 1", command=lambda: App.busy("Clicked Test 1")).grid(row=0, column=0, padx=(0,8))
        ttk.Button(row, text="Test Button 2", command=lambda: App.busy("Clicked Test 2")).grid(row=0, column=1, padx=(0,8))
        ttk.Button(row, text="Open in VS Code", command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=2)
        area = ttk.Frame(self); area.grid(row=2, column=0, sticky="nsew"); self.rowconfigure(2, weight=1)
        self.placeholder(area, "Use this tab to try experimental features and quick checks.")

class AnalysisPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Modeling & Analysis")

        top = ttk.Frame(self); top.grid(row=1, column=0, sticky="ew", pady=(0,8))
        ttk.Button(top, text="Compute from current data", command=self.render).grid(row=0, column=0, sticky="w")
        self.status_lbl = ttk.Label(top, text="", foreground="#6B7280")
        self.status_lbl.grid(row=0, column=1, sticky="w", padx=12)

        self.fig = None
        self.canvas_widget = None
        area = ttk.Frame(self); area.grid(row=2, column=0, sticky="nsew")
        self.rowconfigure(2, weight=1); self.columnconfigure(0, weight=1)
        self.canvas_container = area

    def render(self):
        if not DATA.od or len(DATA.od) < 400:
            self.status_lbl.config(text="Load a CSV with at least a few hundred samples.")
            return
        fe = FeatureEngine(np.array(DATA.od), window=30, fs=1.0)
        fig = fe.fig_sixpack()

        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        self.fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widget = canvas
        self.status_lbl.config(text="Analysis updated.")

# ========================= Main App =========================
class App(tk.Tk):
    _status_var = None

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x800"); self.minsize(1120, 720)

        self._init_style(); self._init_menu()

        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0); root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        sidebar = self._build_sidebar(root); sidebar.grid(row=0, column=0, sticky="nsw")
        self.container = ttk.Frame(root, padding=(12,16,16,16)); self.container.grid(row=0, column=1, sticky="nsew")
        self.container.columnconfigure(0, weight=1); self.container.rowconfigure(0, weight=1)

        self.pages = {
            "Data": DataPage(self.container),
            "Results": ResultsPage(self.container),
            "History": HistoryPage(self.container),
            "Tests": TestsPage(self.container),
            "Analysis": AnalysisPage(self.container),
        }
        for p in self.pages.values(): p.grid(row=0, column=0, sticky="nsew")
        self.show("Results")

        self._build_statusbar()
        self.bind("<Control-1>", lambda e: self.show("Data"))
        self.bind("<Control-2>", lambda e: self.show("Results"))
        self.bind("<Control-3>", lambda e: self.show("History"))
        self.bind("<Control-4>", lambda e: self.show("Analysis"))
        self.bind("<Control-5>", lambda e: self.show("Tests"))

    def _init_style(self):
        self.style = ttk.Style(self)
        try: self.style.theme_use("clam")
        except tk.TclError: pass
        self.style.configure("Sidebar.TFrame", background="#111827")
        self.style.configure("Sidebar.TButton", foreground="white", background="#1F2937")
        self.style.map("Sidebar.TButton", background=[("active", "#374151")])
        self.style.configure("Headline.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("Subhead.TLabel",  font=("Segoe UI", 12, "bold"))
        self.style.configure("Placeholder.TLabel", foreground="#6B7280", background="white")
        self.style.configure("KPI.TLabel", font=("Segoe UI", 10, "bold"))

    def _init_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Project in VS Code", command=lambda: open_in_vscode(os.path.abspath(".")))
        filemenu.add_separator(); filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_sidebar(self, parent):
        bar = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        ttk.Label(bar, text="Wavy Detection", foreground="white", background="#111827",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0,16))
        for name, accel in [("Data","Ctrl+1"),("Results","Ctrl+2"),("History","Ctrl+3"),("Analysis","Ctrl+4"),("Tests","Ctrl+5")]:
            ttk.Button(bar, text=f"{name}    ({accel})", style="Sidebar.TButton",
                       command=lambda n=name: self.show(n)).pack(fill="x", pady=6)
        ttk.Label(bar, text="", background="#111827").pack(expand=True, fill="both")
        ttk.Label(bar, text=f"{APP_VERSION}", foreground="#9CA3AF", background="#111827").pack(anchor="w")
        return bar

    def _build_statusbar(self):
        self._status_var = tk.StringVar(value="Ready")
        bar = ttk.Frame(self); bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self._status_var, padding=8).pack(side="left")
        ttk.Label(bar, text=datetime.now().strftime("%Y-%m-%d"), padding=8).pack(side="right")

    def show(self, page_name: str):
        self.pages[page_name].tkraise(); self.status(f"Showing {page_name}")

    @classmethod
    def status(cls, msg: str):
        if cls._status_var is not None: cls._status_var.set(msg)

    @classmethod
    def busy(cls, msg: str):
        cls.status(msg)

if __name__ == "__main__":
    App().mainloop()
