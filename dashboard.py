# te_qc_gui.py
# Wavy Detection — Prototype Dashboard with History/Trend + Data-driven Gauge
# Built only with Python stdlib + numpy/pandas/matplotlib/sklearn.
# Integrated waviness class overlays (from Streamlit app) for History & Results pages.

import os, math
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
import pickle
from PIL import Image, ImageTk

APP_TITLE   = "Wavy Detection Prototype Dashboard"
APP_VERSION = "v0.5"

# ---- Force the OD column by exact name ----
EXACT_OD_COLUMN = "Tag_value"  # <- put your exact Excel header here
ENFORCE_EXACT_OD = True        # keep True until it plots correctly




DATA_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime", "t_stamp"],
    "od":   ["od", "outer_diameter", "tube_od", "ndc_od_value","ndc_system_ovality_value__tag_value", "tag_value"],
}

# Add this just after DATA_COL_GUESSES
SECONDARY_COL_GUESSES = {
    # time column names we’ll try
    "time": ["ts", "time", "timestamp", "date_time", "datetime", "t_stamp"],
    # likely column names for your “ovality” (or any second metric)
    "val": [
        "ovality", "ovality_value",
        "ndc_system_ovality_value", "ndc_system_ovality_value__tag_value",
        "tag_value", "value"
    ],
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
                # secondary / comparison series
        self.sec_path = None
        self.sec_name = None
        self.sec_ts_dt = []   # parsed timestamps (aligned length to sec_vals)
        self.sec_vals = []    # list[float]
        self.paired_df = None # pandas DataFrame with columns ["od","sec","t"]

        self.model = None
        self.window_size = None

        self.available_sheets = []


    def _read_any_table(self, path: str, sheet = None):
        """Return a pandas DataFrame from CSV/XLS/XLSX (first sheet for Excel)."""
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            if sheet == None:
                return pd.read_excel(path, sheet_name=None, parse_dates=[0])
            else:
                return pd.read_excel(path, sheet_name=[sheet, 'YS_Pullout1_Act_Speed_fpm'], parse_dates=[0])
        
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
    
    def filter_by_speed(self, df_dict):
        speed_df = df_dict['YS_Pullout1_Act_Speed_fpm']
        
        speed_threshold = 1

        mask = speed_df['Tag_value'] > speed_threshold
        filtered_speed = speed_df[mask]
        valid_indices = filtered_speed.index
        
        filtered_dict = {}
        for sheet_name, df in df_dict.items():
            if len(df) == 0:
                continue
                
            filtered_dict[sheet_name] = df.reindex(valid_indices).reset_index(drop=True)
        
        return filtered_dict

    def load_data(self, path: str, app=None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        self.path = path

        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(path)
            self.available_sheets = excel_file.sheet_names
        else:
            self.available_sheets = []

        df_dict = self._read_any_table(path, "NDC_System_OD_Value")
        if df_dict is None or len(df_dict) == 0:
            raise ValueError("Empty file.")
        
        filtered_df_dict = self.filter_by_speed(df_dict)

        self.od = filtered_df_dict['NDC_System_OD_Value']['Tag_value'].tolist()
        self.ts = filtered_df_dict['NDC_System_OD_Value']['t_stamp'].astype(str).tolist()
        self.ts_dt = pd.to_datetime(filtered_df_dict['NDC_System_OD_Value']['t_stamp'], errors='coerce').tolist()

        try:
            v = self.od
            App.status(f"Using time='{'t_stamp'}', OD='{'Tag_value'}' • rows={len(v)} "
                    f"• min={v.min():.6g}, max={v.max():.6g}, mean={v.mean():.6g}")
        except Exception:
            pass

        self.last_loaded_rows = len(self.od)
        self.path = path

        try:
            App.status(f"Using time='{'t_stamp'}', OD='{'Tag_value'}'. Rows kept: {self.last_loaded_rows}.")
        except Exception:
            pass

        if self.model is not None and app is not None:
            self.auto_classify(window_size=self.window_size)
            if hasattr(app, 'pages') and 'Analysis' in app.pages:
                analysis_page = app.pages['Analysis']
                analysis_page.update_confidence_timeline()

            # Map labels -> an NG risk in [0..1] (tune if you like)
    CLASS_TO_RISK = {
        "STEADY":        0.05,
        "MILD_WAVE":     0.40,
        "STRONG_WAVE":   0.90,
        "DRIFT":         0.80,
        "BURSTY_NOISY":  0.85,
        "UNCERTAIN":     0.50,
    }

    def _align_series(self, df_main, tcol_main, ycol_main, df_sec, tcol_sec, ycol_sec):
        """
        Build an inner-joined (nearest) table of OD and secondary values by time.
        Returns a DataFrame with columns: ['t','od','sec'].
        """
        # Prepare main
        m = pd.DataFrame({
            "t": pd.to_datetime(df_main[tcol_main], errors="coerce"),
            "od": self._smart_to_numeric(df_main[ycol_main]),
        }).dropna()
        # Prepare secondary
        s = pd.DataFrame({
            "t": pd.to_datetime(df_sec[tcol_sec], errors="coerce"),
            "sec": self._smart_to_numeric(df_sec[ycol_sec]),
        }).dropna()

        # Remove timezone so we can compare
        for col in ["t"]:
            if hasattr(m[col], "dt"):
                try: m[col] = m[col].dt.tz_convert(None)
                except: m[col] = m[col].dt.tz_localize(None)
            if hasattr(s[col], "dt"):
                try: s[col] = s[col].dt.tz_convert(None)
                except: s[col] = s[col].dt.tz_localize(None)

        m = m.sort_values("t")
        s = s.sort_values("t")

        # If both look evenly sampled, use an asof join with a small tolerance
        # Tolerance = median step of the faster stream (or 1s fallback).
        tol = pd.Timedelta(seconds=1)
        if len(m) >= 3:
            dtm = (m["t"].diff().dropna().median() or pd.Timedelta(seconds=1))
            tol = max(tol, dtm)
        if len(s) >= 3:
            dts = (s["t"].diff().dropna().median() or pd.Timedelta(seconds=1))
            tol = max(tol, dts)

        paired = pd.merge_asof(m, s, on="t", direction="nearest", tolerance=tol)
        paired = paired.dropna().reset_index(drop=True)
        return paired[["t","od","sec"]]
    
    def load_secondary_sheet(self, sheet_name: str):
        if not self.path:
            raise ValueError("Load the main data file first.")
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        
        ext = os.path.splitext(self.path.lower())[1]
        if ext not in [".xlsx", ".xls"]:
            raise ValueError("Secondary sheet loading only works with Excel files.")
        
        df_dict = pd.read_excel(self.path, sheet_name=[sheet_name, 'YS_Pullout1_Act_Speed_fpm'])
        
        df_sec = df_dict[sheet_name]
        if df_sec is None or df_sec.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty.")
        
        filtered_dict = self.filter_by_speed(df_dict)
        df_sec_filtered = filtered_dict[sheet_name]

        cols_s = list(df_sec_filtered.columns)
        tcol_s = pick(cols_s, SECONDARY_COL_GUESSES["time"]) or "t_stamp"
        ycol_s = pick(cols_s, SECONDARY_COL_GUESSES["val"]) or "Tag_value"
        
        if tcol_s not in cols_s:
            raise ValueError(f"Could not find time column in sheet '{sheet_name}'. Columns: {cols_s}")
        if ycol_s not in cols_s:
            raise ValueError(f"Could not find value column in sheet '{sheet_name}'. Columns: {cols_s}")

        df_dict_od = pd.read_excel(self.path, sheet_name=['NDC_System_OD_Value', 'YS_Pullout1_Act_Speed_fpm'])
        filtered_dict_od = self.filter_by_speed(df_dict_od)
        df_main_filtered = filtered_dict_od['NDC_System_OD_Value']
        
        tcol_m = "t_stamp"
        ycol_m = "Tag_value"

        paired = self._align_series(df_main_filtered, tcol_m, ycol_m, 
                                    df_sec_filtered, tcol_s, ycol_s)
        if paired.empty:
            raise ValueError(f"No overlapping timestamps between OD and '{sheet_name}' after speed filtering.")

        self.paired_df = paired
        self.sec_path = self.path
        self.sec_name = sheet_name
        self.sec_ts_dt = paired["t"].tolist()
        self.sec_vals = paired["sec"].tolist()

        App.status(f"Secondary loaded & aligned: {sheet_name} • paired rows={len(self.paired_df)} (speed filtered)")

    def corr_stats(self, max_lag_samples: int = 300):
        """
        Compute basic correlation stats between OD and secondary series.
        Returns dict with: n, pearson_r, best_lag, r_at_best_lag.
        """
        if self.paired_df is None or self.paired_df.empty:
            return {"n": 0, "pearson_r": np.nan, "best_lag": 0, "r_at_best_lag": np.nan}

        x = self.paired_df["od"].to_numpy(dtype=float)
        y = self.paired_df["sec"].to_numpy(dtype=float)
        n = min(len(x), len(y))
        if n < 3:
            return {"n": n, "pearson_r": np.nan, "best_lag": 0, "r_at_best_lag": np.nan}

        # plain Pearson (no lag)
        r0 = float(np.corrcoef(x, y)[0,1])

        # best lag (shift y relative to x)
        best_r, best_k = r0, 0
        K = min(max_lag_samples, n-2)
        for k in range(1, K+1):
            r_pos = float(np.corrcoef(x[k:], y[:-k])[0,1])
            if r_pos > best_r: best_r, best_k = r_pos, +k
            r_neg = float(np.corrcoef(x[:-k], y[k:])[0,1])
            if r_neg > best_r: best_r, best_k = r_neg, -k

        return {"n": n, "pearson_r": r0, "best_lag": best_k, "r_at_best_lag": best_r}

    def _paired_ok(self):
        return (self.paired_df is not None) and (not self.paired_df.empty)

    def estimate_sample_period_s(self, df=None):
        """Median seconds between samples from a time column."""
        if df is None:
            if not self._paired_ok(): return None
            df = self.paired_df
        t = pd.to_datetime(df["t"], errors="coerce").dropna()
        if len(t) < 3: return None
        dt = t.diff().dropna().median()
        try:
            return float(dt.total_seconds())
        except Exception:
            return None

    def lag_corr_curve(self, max_lag_samples=300):
        """
        Return lags (in samples) and Pearson r at each lag (y shifted).
        lags: array from -K..+K, r: same length.
        """
        if not self._paired_ok(): return np.array([]), np.array([])
        x = self.paired_df["od"].to_numpy(dtype=float)
        y = self.paired_df["sec"].to_numpy(dtype=float)
        n = min(len(x), len(y))
        if n < 5: return np.array([]), np.array([])
        K = int(min(max_lag_samples, n - 3))
        lags = np.arange(-K, K+1, dtype=int)
        r = np.zeros_like(lags, dtype=float)
        for i, k in enumerate(lags):
            if k < 0:   # y leads (shift y forward)
                r[i] = np.corrcoef(x[:k], y[-k:])[0,1]
            elif k > 0: # x leads
                r[i] = np.corrcoef(x[k:], y[:-k])[0,1]
            else:
                r[i] = np.corrcoef(x, y)[0,1]
        return lags, r

    def rolling_corr(self, win_samples=200, step=10):
        """
        Rolling Pearson r over paired series.
        Returns arrays (t_mid, r_roll). t_mid are midpoint timestamps of each window.
        """
        if not self._paired_ok(): return np.array([]), np.array([])
        df = self.paired_df
        x = df["od"].to_numpy(dtype=float)
        y = df["sec"].to_numpy(dtype=float)
        t = pd.to_datetime(df["t"], errors="coerce").to_numpy()
        n = len(df)
        if n < max(10, win_samples): return np.array([]), np.array([])
        mids, rr = [], []
        for i0 in range(0, n - win_samples + 1, step):
            i1 = i0 + win_samples
            segx, segy = x[i0:i1], y[i0:i1]
            if np.std(segx) < 1e-12 or np.std(segy) < 1e-12:
                r = np.nan
            else:
                r = float(np.corrcoef(segx, segy)[0,1])
            mids.append(t[i0 + win_samples//2])
            rr.append(r)
        return np.array(mids), np.array(rr)

    def current_timestamp(self):
        """Timestamp of the latest OD sample (pd.Timestamp or None)."""
        if not self.ts_dt:
            return None
        return self.ts_dt[-1]

    def current_class(self):
        if not self.classes:
            return None, None
        return self.classes[-1]["label"], self.classes[-1]["risk"]

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

    def extract_features(self, window_data):
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        # div by zero
        if mean_val == 0:
            mean_val = 1e-10
        
        # features are normalized to the mean here to take into account spans of multiple different tubing target sizes
        features = {
            'coef_variation': std_val / mean_val,  # coefficient of variation ak relative std
            'relative_range': np.ptp(window_data) / mean_val,  # range relative to mean
            'normalized_variance': np.var(window_data) / (mean_val ** 2),
            'peak_to_peak_ratio': np.ptp(window_data) / np.abs(mean_val),
            'relative_max_deviation': (np.max(window_data) - mean_val) / mean_val,
            'relative_min_deviation': (mean_val - np.min(window_data)) / mean_val,
            # differences between consecutive points, indicates smoothness
            'mean_abs_diff': np.mean(np.abs(np.diff(window_data))) / mean_val,
            'max_abs_diff': np.max(np.abs(np.diff(window_data))) / mean_val,
        }
        
        return features
    
    def get_label_from_risk_prob(self, risk):
        if   risk < 0.25: return "STEADY"
        elif risk < 0.45: return "MILD_WAVE"
        elif risk < 0.65: return "DRIFT"
        elif risk < 0.80: return "BURSTY_NOISY"
        else:             return "STRONG_WAVE"

    def auto_classify(self, window_size=60):
        if self.model is None:
            App.status("No model selected. Please select a model first.")
            return
        
        # Clear existing classes
        self.classes = []
        
        num_windows = len(self.od) // window_size

        # Collect all features first
        features_list = []
        window_metadata = []  # Store start/end indices for each window
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = self.od[start_idx:end_idx]
            
            # Extract features as a dictionary
            features_dict = self.extract_features(window)
            features_list.append(features_dict)
            window_metadata.append((start_idx, end_idx))
        
        # Convert to DataFrame (same as training)
        X = pd.DataFrame(features_list)
        
        # Scale all features at once (same as training)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        
        # Predict for all windows
        probas = self.model.predict_proba(X)
        
        # Create class segments
        for i, (start_idx, end_idx) in enumerate(window_metadata):
            chatter_confidence = probas[i][1]  # Probability of class 1 (chatter/bad)
            
            self.classes.append({
                "start": self.ts[start_idx],
                "end": self.ts[end_idx],
                "label": self.get_label_from_risk_prob(chatter_confidence),
                "i0": start_idx,
                "i1": end_idx,
                "risk": chatter_confidence
            })

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
    # 111111111
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
            # Store image references to prevent garbage collection
            if not hasattr(self, '_overlay_images'):
                self._overlay_images = []
            self._overlay_images.clear()
            
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

                x0 = X(i0)
                x1 = X(i1)
                
                # Get color and create semi-transparent overlay
                color = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                rgb = self.canvas.winfo_rgb(color)
                # winfo_rgb returns 16-bit values (0-65535), convert to 8-bit (0-255)
                r = rgb[0] >> 8
                g = rgb[1] >> 8
                b = rgb[2] >> 8
                alpha = 80  # transparency (0=transparent, 255=opaque)
                
                # Create RGBA image with proper dimensions
                width = int(x1 - x0)
                height = int(y1 - y0)
                
                if width > 0 and height > 0:
                    image = Image.new('RGBA', (width, height), (r, g, b, alpha))
                    photo = ImageTk.PhotoImage(image)
                    self._overlay_images.append(photo)  # Keep reference!
                    
                    # Use x0, y0 with anchor='nw' to position at top-left
                    self.canvas.create_image(x0, y0, image=photo, anchor='nw')
                
                # Draw label
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

        ttk.Button(controls, text="Load XLSX…", command=self.load_xlsx).grid(row=0, column=0, sticky="w", padx=(0,8))
        
        ttk.Label(controls, text="Compare to:").grid(row=0, column=8, sticky="w", padx=(0,4))
        self.sheet_var = tk.StringVar(value="Select sheet...")
        self.sheet_dropdown = ttk.Combobox(controls, textvariable=self.sheet_var, state="disabled", width=25)
        self.sheet_dropdown.grid(row=0, column=9, sticky="w", padx=(0,8))
        self.sheet_dropdown.bind('<<ComboboxSelected>>', self.on_sheet_select)
        
        ttk.Button(controls, text="Show Correlation", command=self.show_corr).grid(row=0, column=10, sticky="w", padx=(0,8))


        self.info = ttk.Label(self, text="No file loaded.", foreground="#6B7280")
        self.info.grid(row=2, column=0, sticky="w")

        table_area = ttk.Frame(self); table_area.grid(row=3, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(3, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

    def load_xlsx(self):
        path = filedialog.askopenfilename(
            title="Select XLSX file",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not path: return
        try:
            app_instance = self.winfo_toplevel()
            DATA.load_data(path, app=app_instance)
            self.info.config(text=f"Loaded: {os.path.basename(path)}  •  rows={len(DATA.od)}")
            
            excluded = ['NDC_System_OD_Value', 'YS_Pullout1_Act_Speed_fpm']
            available = [s for s in DATA.available_sheets if s not in excluded]
            
            if available:
                self.sheet_dropdown['values'] = available
                self.sheet_dropdown['state'] = 'readonly'
                self.sheet_var.set("Select sheet...")
            else:
                self.sheet_dropdown['values'] = []
                self.sheet_dropdown['state'] = 'disabled'
                self.sheet_var.set("No other sheets")

            app_instance = self.winfo_toplevel()
            if hasattr(app_instance, 'pages') and 'Model' in app_instance.pages:
                app_instance.pages['Model'].reset_confidence_plot()
            
            App.status("Data loaded. History & gauge now using real data.")
        except Exception as e:
            messagebox.showerror("Load Data failed", str(e))
            App.status("Data load failed")

    def on_sheet_select(self, event):
        selected_sheet = self.sheet_var.get()
        if selected_sheet and selected_sheet not in ["Select sheet...", "No other sheets"]:
            try:
                DATA.load_secondary_sheet(selected_sheet)
                self.info.config(text=f"{self.info.cget('text')}  •  comparing to '{selected_sheet}' (paired={len(DATA.paired_df)})")
                App.status(f"Secondary sheet loaded: {selected_sheet}")
            except Exception as e:
                messagebox.showerror("Load Secondary failed", str(e))
                App.status("Load secondary failed")
                self.sheet_var.set("Select sheet...")

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

    def show_corr(self):
        if DATA.paired_df is None or DATA.paired_df.empty:
            messagebox.showinfo("Correlation", "Select a secondary sheet first from the dropdown.")
            return
        CorrelationWindow(self)


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

class ModelPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Select a model and window size")
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)
        
        left_frame = ttk.Frame(self)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        
        ttk.Label(left_frame, text="Select Model:", style="Subhead.TLabel").pack(anchor="w", pady=(0, 8))
        
        self.models = {}
        self.load_models()

        self.cb = ttk.Combobox(left_frame, values=sorted(list(self.models.keys())), height=30, width=40)
        self.cb.set("Pick a model")
        self.cb.pack(pady=(0, 16))
        self.cb.bind('<<ComboboxSelected>>', self.on_model_select)
        
        ttk.Button(left_frame, text="Update Likelihood Plot", command=self.update_confidence_plot).pack(pady=(8, 0))
        
        right_frame = ttk.Frame(self)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        
        ttk.Label(right_frame, text="Average likelihood of chatter being detected in entire dataset (all windows), by each model over different window sizes\nHigher likelihood = chatter likely present | Lower likelihood = chatter unlikely | Middle likelihood = unsure prediction", 
                 style="Subhead.TLabel").pack(anchor="w", pady=(0, 8))
        
        self.fig_conf = Figure(figsize=(8, 5), dpi=100)
        self.ax_conf = self.fig_conf.add_subplot(111)
        self.canvas_conf = FigureCanvasTkAgg(self.fig_conf, master=right_frame)
        self.canvas_conf.get_tk_widget().pack(fill="both", expand=True)
        
        self.ax_conf.text(0.5, 0.5, "Load data and click 'Update Likelihood Plot'\nto see model predictions", 
                         ha="center", va="center", fontsize=12)
        self.ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()

    def on_model_select(self, event):
        DATA.model = self.models[self.cb.get()]
        DATA.window_size = int(self.cb.get().split('Window Size ')[1].rstrip(')'))
        if DATA.od:
            DATA.auto_classify(DATA.window_size)
            app_instance = self.winfo_toplevel()
            # if hasattr(app_instance, 'pages') and 'Results' in app_instance.pages:
            #     results_page = app_instance.pages['Results']
            #     results_page._tick()
            if hasattr(app_instance, 'pages') and 'Analysis' in app_instance.pages:
                analysis_page = app_instance.pages['Analysis']
                analysis_page.update_confidence_timeline()
            
            App.status(f"Model changed and classifications updated")

    def update_confidence_plot(self):
        if not DATA.od or len(DATA.od) < 100:
            messagebox.showinfo("No Data", "Please load data first (at least 100 samples).")
            return
        
        App.status("Computing confidence curves... this may take a moment")
        
        # Group models by type and window size
        model_types = {}  # {model_type: {window_size: model}}
        for model_name, model in self.models.items():
            # Parse model name: "Model Type (Window Size XX)"
            parts = model_name.rsplit(' (Window Size ', 1)
            if len(parts) == 2:
                model_type = parts[0]
                window_size = int(parts[1].rstrip(')'))
                
                if model_type not in model_types:
                    model_types[model_type] = {}
                model_types[model_type][window_size] = model
        
        # Compute average confidence for each model type at each window size
        results = {}  # {model_type: ([window_sizes], [avg_confidences])}
        
        for model_type, ws_dict in model_types.items():
            window_sizes = []
            avg_confidences = []
            
            for ws in sorted(ws_dict.keys()):
                model = ws_dict[ws]
                
                num_windows = len(DATA.od) // ws
                if num_windows < 1:
                    continue
                
                features_list = []
                for i in range(num_windows):
                    start_idx = i * ws
                    end_idx = start_idx + ws
                    if end_idx > len(DATA.od):
                        break
                    window = DATA.od[start_idx:end_idx]
                    features_dict = DATA.extract_features(window)
                    features_list.append(features_dict)
                
                if not features_list:
                    continue
                
                X = pd.DataFrame(features_list)
                probas = model.predict_proba(X)
                
                avg_conf = np.mean(probas[:, 1]) * 100 
                
                window_sizes.append(ws)
                avg_confidences.append(avg_conf)
            
            if window_sizes:
                results[model_type] = (window_sizes, avg_confidences)
        
        self.ax_conf.clear()
        
        colors = ['#2563EB', '#DC2626', '#16A34A', '#F59E0B']
        markers = ['o', 's', '^', 'd']
        
        for idx, (model_type, (ws, confs)) in enumerate(sorted(results.items())):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            self.ax_conf.plot(ws, confs, marker=marker, linewidth=2, markersize=6,
                            label=model_type, color=color)
        
        self.ax_conf.set_xlabel('Window Size (samples)', fontsize=11)
        self.ax_conf.set_ylabel('Chatter Likelihood (%)', fontsize=11)
        self.ax_conf.set_title('Average Chatter Likelihood vs Window Size', 
                              fontsize=12, fontweight='bold')
        self.ax_conf.legend(loc='best', fontsize=9)
        self.ax_conf.grid(True, alpha=0.3)
        self.ax_conf.set_ylim([0, 105])
        
        self.ax_conf.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_conf.text(self.ax_conf.get_xlim()[1], 50, ' Decision boundary', 
                         va='center', fontsize=9, color='gray')
        
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()
        
        App.status(f"Confidence plot updated with {len(results)} model types")

    def load_models(self):
        all_model_files = os.listdir("models")
        for model_file in all_model_files:
            if not model_file.endswith('.pkl') or model_file.startswith('scaler_'):
                continue
                
            model_path = os.path.join("models", model_file)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                name_parts = model_file.split("_")
                match name_parts[1]:
                    case 'logit':
                        model_name = f"Logistic Regression (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case 'rf':
                        model_name = f"Random Forest (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case 'svm':
                        model_name = f"SVM (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case 'xgboost':
                        model_name = f"XGBoost (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case _:
                        self.models[model_file] = model

    def reset_confidence_plot(self):
        self.ax_conf.clear()
        self.ax_conf.text(0.5, 0.5, "Load data and click 'Update Likelihood Plot'\nto see model predictions", 
                         ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()
        App.status("Confidence plot reset")

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
        self.headline("Results")

        tools = ttk.Frame(self); tools.grid(row=1, column=0, sticky="ew", pady=(0,12))
        # ttk.Button(tools, text="Predict Latest", command=self.predict_latest).grid(row=0, column=0, sticky="w", padx=(0,8))
        # ttk.Button(tools, text="Export Report", command=lambda: App.busy("Export report (todo)")).grid(row=0, column=1, sticky="w", padx=(0,8))

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

        self.pred_label = ttk.Label(right_top, text="Predicted class: —", style="KPI.TLabel")
        self.pred_label.pack(anchor="w", pady=(6, 0))
        self.pred_conf  = ttk.Label(right_top, text="Confidence: —", style="KPI.TLabel")
        self.pred_conf.pack(anchor="w")

        self.after(1000, self._tick)


    def _tick(self):
        pct = 50
        
        if DATA.od and DATA.classes:
            lbl, class_risk = DATA.current_class()
            if class_risk is not None:
                pct = class_risk * 100.0
                self.pred_label.config(text=f"Predicted class: {lbl}")
                self.pred_conf.config(text=f"Confidence: {pct:0.1f}%")
            else:
                pct = 0
                self.pred_label.config(text="Error")
                self.pred_conf.config(text=f"Error")
        elif DATA.od:
            # data loaded but no model selected yet
            pct = 50 + 15 * math.sin(datetime.now().timestamp()/2.0)
            self.pred_label.config(text="Predicted class: —")
            self.pred_conf.config(text="Please select a model")
        else:
            # no data - demo mode
            pct = 50 + 15 * math.sin(datetime.now().timestamp()/2.0)
            self.pred_label.config(text="Predicted class: —")
            self.pred_conf.config(text="Please import data and select a model")

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

class AnalysisPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Modeling & Analysis")

        top = ttk.Frame(self); top.grid(row=1, column=0, sticky="ew", pady=(0,8))
        self.status_lbl = ttk.Label(top, text="", foreground="#6B7280")
        self.status_lbl.grid(row=0, column=0, sticky="w", padx=12)

        area = ttk.Frame(self); area.grid(row=2, column=0, sticky="nsew")
        self.rowconfigure(2, weight=1); self.columnconfigure(0, weight=1)
        
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self._overlay_images = []
        
        self.ax.text(0.5, 0.5, "Load data and select a model to see predictions", 
                    ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

    def update_confidence_timeline(self):
        if not DATA.od or len(DATA.od) < 100:
            messagebox.showinfo("No Data", "Please load data first (at least 100 samples).")
            return
        
        if DATA.model is None:
            messagebox.showinfo("No Model", "Please select a model first.")
            return
        
        App.status("Computing confidence timeline...")
        
        self._overlay_images.clear()
        self.ax.clear()
        
        ws = DATA.window_size
        num_windows = len(DATA.od) // ws
        
        if num_windows < 1:
            self.ax.text(0.5, 0.5, "Not enough data for the selected window size", 
                        ha="center", va="center", fontsize=12)
            self.ax.axis("off")
            self.canvas.draw()
            return
        
        confidences = []
        window_times = []
        
        if not DATA.ts_dt:
            DATA.ts_dt = pd.to_datetime(DATA.ts, errors='coerce').tolist()
            if not DATA.ts_dt or all(pd.isna(t) for t in DATA.ts_dt):
                DATA.ts_dt = [pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(len(DATA.od))]
        
        for i in range(num_windows):
            start_idx = i * ws
            end_idx = start_idx + ws
            window = DATA.od[start_idx:end_idx]
            
            mid_idx = start_idx + ws // 2
            if mid_idx < len(DATA.ts_dt) and DATA.ts_dt[mid_idx] is not None:
                window_times.append(DATA.ts_dt[mid_idx])
            elif len(DATA.ts_dt) > 0:
                valid_ts = [t for t in DATA.ts_dt if t is not None and pd.notna(t)]
                if valid_ts:
                    window_times.append(valid_ts[-1])
                else:
                    window_times.append(pd.Timestamp.now())
            else:
                window_times.append(pd.Timestamp.now())
            
            features_dict = DATA.extract_features(window)
            X = pd.DataFrame([features_dict])
            
            proba = DATA.model.predict_proba(X)
            chatter_conf = proba[0][1] * 100
            confidences.append(chatter_conf)
        
        self.ax.plot(window_times, confidences, linewidth=2, color='#2563EB', 
                    label='Chatter Likelihood (%)', zorder=10)
        
        self.ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, 
                       alpha=0.5, label='Decision Boundary', zorder=5)
        
        if DATA.classes:
            y_min, y_max = self.ax.get_ylim()
            
            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                
                start_time = seg["start"]
                end_time = seg["end"]
                
                color = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                
                self.ax.axvspan(start_time, end_time, 
                              facecolor=color, alpha=0.25, 
                              linewidth=0, zorder=1)
        
        self.ax.set_xlabel('Time', fontsize=11)
        self.ax.set_ylabel('Chatter Likelihood (%)', fontsize=11)
        self.ax.set_title(f'Chatter Confidence in each Window over Time (Window Size: {ws} samples)', 
                         fontsize=12, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3, zorder=0)
        self.ax.set_ylim([0, 105])
        
        legend_elements = []
        for name, color in CLASS_COLORS.items():
            if name in VISIBLE_CLASSES:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, alpha=0.25, label=name))
        
        if legend_elements:
            second_legend = self.ax.legend(handles=legend_elements, 
                                          loc='upper left', 
                                          fontsize=8, 
                                          title='Wave Classes')
            self.ax.add_artist(second_legend)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.status_lbl.config(text=f"Timeline updated: {num_windows} windows analyzed")
        App.status(f"Confidence timeline computed for {num_windows} windows")

class CorrelationWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("OD ↔ Secondary Correlation")
        self.minsize(980, 720)

        if DATA.paired_df is None or DATA.paired_df.empty:
            ttk.Label(self, text="Load a secondary file first.", padding=12).pack()
            return

        # === Top stats / controls ===
        top = ttk.Frame(self, padding=12)
        top.pack(side="top", fill="x")

        stats = DATA.corr_stats()
        rtxt = f"r = {stats['pearson_r']:.3f}  |  best lag: {stats['best_lag']} samples (r={stats['r_at_best_lag']:.3f})"
        sign = "POSITIVE" if stats["pearson_r"] >= 0 else "NEGATIVE"
        color = "#16A34A" if stats["pearson_r"] >= 0 else "#DC2626"

        ttk.Label(top, text=f"Paired rows: {stats['n']}", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,20))
        ttk.Label(top, text=rtxt).grid(row=0, column=1, sticky="w", padx=(0,20))
        badge = ttk.Label(top, text=f"Tendency: {sign}", foreground=color, font=("Segoe UI", 10, "bold"))
        badge.grid(row=0, column=2, sticky="w")

        # Controls
        ctrl = ttk.Frame(top)
        ctrl.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8,0))
        ttk.Label(ctrl, text="Max lag (samples):").grid(row=0, column=0, sticky="w")
        self.maxlag_var = tk.IntVar(value=300)
        ttk.Entry(ctrl, textvariable=self.maxlag_var, width=8).grid(row=0, column=1, sticky="w", padx=(6,16))
        ttk.Label(ctrl, text="Rolling window (samples):").grid(row=0, column=2, sticky="w")
        self.win_var = tk.IntVar(value=200)
        ttk.Entry(ctrl, textvariable=self.win_var, width=8).grid(row=0, column=3, sticky="w", padx=(6,16))
        ttk.Button(ctrl, text="Update plots", command=self._refresh_all).grid(row=0, column=4)

        # === Tabs ===
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True, padx=12, pady=12)

        self.tab_overlay = ttk.Frame(nb); nb.add(self.tab_overlay, text="Time Overlay")
        self.tab_scatter = ttk.Frame(nb); nb.add(self.tab_scatter, text="Scatter & Density")
        self.tab_lag     = ttk.Frame(nb); nb.add(self.tab_lag,     text="Corr vs Lag")
        self.tab_roll    = ttk.Frame(nb); nb.add(self.tab_roll,    text="Rolling Corr")

        # Matplotlib canvases
        self.fig_overlay = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_overlay  = self.fig_overlay.add_subplot(111)
        self.cv_overlay  = FigureCanvasTkAgg(self.fig_overlay, master=self.tab_overlay)
        self.cv_overlay.get_tk_widget().pack(fill="both", expand=True)

        self.fig_scatter = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_scatter  = self.fig_scatter.add_subplot(111)
        self.cv_scatter  = FigureCanvasTkAgg(self.fig_scatter, master=self.tab_scatter)
        self.cv_scatter.get_tk_widget().pack(fill="both", expand=True)

        self.fig_lag = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_lag  = self.fig_lag.add_subplot(111)
        self.cv_lag  = FigureCanvasTkAgg(self.fig_lag, master=self.tab_lag)
        self.cv_lag.get_tk_widget().pack(fill="both", expand=True)

        self.fig_roll = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_roll  = self.fig_roll.add_subplot(111)
        self.cv_roll  = FigureCanvasTkAgg(self.fig_roll, master=self.tab_roll)
        self.cv_roll.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_all()

    # ---- drawing helpers ----
    def _refresh_all(self):
        self._draw_overlay()
        self._draw_scatter()
        self._draw_lag()
        self._draw_rolling()

    def _draw_overlay(self):
        df = DATA.paired_df.copy()
        df["od_z"]  = (df["od"]  - df["od"].mean())  / (df["od"].std(ddof=0)  + 1e-12)
        df["sec_z"] = (df["sec"] - df["sec"].mean()) / (df["sec"].std(ddof=0) + 1e-12)

        self.ax_overlay.clear()
        t = pd.to_datetime(df["t"], errors="coerce")
        self.ax_overlay.plot(t, df["od_z"],  linewidth=1.5, label="OD (z-score)")
        self.ax_overlay.plot(t, df["sec_z"], linewidth=1.5, label="Secondary (z-score)")
        self.ax_overlay.axhline(0, linewidth=0.8, color="#999999")
        self.ax_overlay.set_title("Time Overlay (z-scored)")
        self.ax_overlay.set_xlabel("time"); self.ax_overlay.set_ylabel("z-score")
        self.ax_overlay.legend(loc="upper right")
        self.fig_overlay.tight_layout()
        self.cv_overlay.draw()

    def _draw_scatter(self):
        df = DATA.paired_df
        x = df["od"].to_numpy(dtype=float)
        y = df["sec"].to_numpy(dtype=float)

        self.ax_scatter.clear()
        # Use hexbin for “density”; easy to eyeball slope & structure
        hb = self.ax_scatter.hexbin(x, y, gridsize=40, bins="log")
        self.fig_scatter.colorbar(hb, ax=self.ax_scatter, fraction=0.046, pad=0.04, label="log density")

        # Fit line y = a*x + b
        if len(x) >= 2:
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 200)
            self.ax_scatter.plot(xx, a*xx + b, linewidth=2, label=f"Fit: y={a:.3f}x+{b:.3f}")

        # Pearson r
        r = float(np.corrcoef(x, y)[0,1]) if len(x) > 1 else np.nan
        self.ax_scatter.set_title(f"Scatter & Density  (r={r:.3f})")
        self.ax_scatter.set_xlabel("OD"); self.ax_scatter.set_ylabel("Secondary")
        self.ax_scatter.legend(loc="best")
        self.fig_scatter.tight_layout()
        self.cv_scatter.draw()

    def _draw_lag(self):
        maxlag = max(5, int(self.maxlag_var.get()))
        lags, r = DATA.lag_corr_curve(max_lag_samples=maxlag)
        self.ax_lag.clear()
        if lags.size:
            self.ax_lag.plot(lags, r, linewidth=2)
            self.ax_lag.axhline(0, linewidth=0.8, color="#999999")
            # annotate best
            k = int(lags[np.nanargmax(r)])
            rmax = float(np.nanmax(r))
            self.ax_lag.axvline(k, linestyle="--", linewidth=1.2)
            self.ax_lag.set_title(f"Correlation vs Lag (best: {k} samples, r={rmax:.3f})")
            self.ax_lag.set_xlabel("lag (samples, + = OD leads)"); self.ax_lag.set_ylabel("Pearson r")
        else:
            self.ax_lag.text(0.5, 0.5, "Not enough paired data.", ha="center", va="center")
            self.ax_lag.axis("off")
        self.fig_lag.tight_layout()
        self.cv_lag.draw()

    def _draw_rolling(self):
        win = max(20, int(self.win_var.get()))
        t, r = DATA.rolling_corr(win_samples=win, step=max(5, win//10))
        self.ax_roll.clear()
        if r.size:
            self.ax_roll.plot(t, r, linewidth=2)
            self.ax_roll.axhline(0, linewidth=0.8, color="#999999")
            self.ax_roll.set_title(f"Rolling Correlation (window={win} samples)")
            self.ax_roll.set_xlabel("time"); self.ax_roll.set_ylabel("Pearson r")
        else:
            self.ax_roll.text(0.5, 0.5, "Not enough paired data.", ha="center", va="center")
            self.ax_roll.axis("off")
        self.fig_roll.tight_layout()
        self.cv_roll.draw()



# ========================= Main App =========================
class App(tk.Tk):
    _status_var = None

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x850"); self.minsize(1120, 720)

        self._init_style(); self._init_menu()

        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0); root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        sidebar = self._build_sidebar(root); sidebar.grid(row=0, column=0, sticky="nsw")
        self.container = ttk.Frame(root, padding=(12,16,16,16)); self.container.grid(row=0, column=1, sticky="nsew")
        self.container.columnconfigure(0, weight=1); self.container.rowconfigure(0, weight=1)

        self.pages = {
            "Data": DataPage(self.container),
            "Model": ModelPage(self.container),
            "Results": ResultsPage(self.container),
            "History": HistoryPage(self.container),
            "Analysis": AnalysisPage(self.container),
        }
        for p in self.pages.values(): p.grid(row=0, column=0, sticky="nsew")
        self.show("Results")

        self._build_statusbar()
        self.bind("<Control-1>", lambda e: self.show("Data"))
        self.bind("<Control-2>", lambda e: self.show("Results"))
        self.bind("<Control-3>", lambda e: self.show("History"))
        self.bind("<Control-4>", lambda e: self.show("Analysis"))

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
        filemenu.add_separator(); filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_sidebar(self, parent):
        bar = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        ttk.Label(bar, text="Wavy Detection", foreground="white", background="#111827",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0,16))
        for name, accel in [("Data","Ctrl+1"),("Model","Ctrl+2"),("Results","Ctrl+3"),("History","Ctrl+4"),("Analysis","Ctrl+5")]:
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
