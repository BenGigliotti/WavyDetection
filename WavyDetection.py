# te_qc_gui.py
# Wavy Detection — Prototype Dashboard with History/Trend + Data-driven Gauge
# Excel-first, with colored waviness bands in Results + History.

import os, math, random, subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np, pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Optional (only used in the Analysis tab "six-pack")
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

APP_TITLE   = "Wavy Detection Prototype Dashboard"
APP_VERSION = "v0.4 (Excel + colored bands)"

DATA_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime"],
    "od":   ["od", "outer_diameter", "tube_od", "ndc_od_value", "Tag_value"],
}

# ----- Color map used for shading bands -----
CLASS_COLORS = {
    "STEADY":       "#4CAF50",   # green
    "MILD_WAVE":    "#FFB300",   # amber
    "STRONG_WAVE":  "#E53935",   # red
    "DRIFT":        "#2196F3",   # blue
    "BURSTY_NOISY": "#8E24AA",   # purple
    "UNCERTAIN":    "#9E9E9E",   # grey
}

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
    low = [str(c).lower() for c in colnames]
    for alias in candidates:
        if alias.lower() in low:
            return colnames[low.index(alias.lower())]
    return None

def rolling_mad(x):
    m = np.median(x)
    return np.mean(np.abs(x - m))

# ========================= Waviness classifier (sample-based windows) =========================
def _win_features(y, t):
    # y: np.array window, t: np.array 0..T (seconds or samples)
    if len(y) < 8:
        return dict(ptp_amp=0.0, std=0.0, max_acf=0.0, dom_power_ratio=0.0, slope=0.0)
    coef = np.polyfit(t, y, 1)
    trend = np.polyval(coef, t)
    yd = y - trend
    ptp_amp = float(np.max(y) - np.min(y))
    std = float(np.std(y))
    z = yd - np.mean(yd)
    if np.allclose(z, 0):
        max_acf = 0.0
    else:
        acf = np.correlate(z, z, mode="full")[len(z)-1:]
        acf = acf / (acf[0] if acf[0] != 0 else 1.0)
        max_acf = float(np.max(acf[1:max(2, len(z)//2)])) if len(z) > 2 else 0.0
    Y = np.abs(np.fft.rfft(yd - np.mean(yd)))
    dom_power_ratio = float(Y[1:].max() / Y.sum()) if Y.sum() > 0 and len(Y) > 1 else 0.0
    return dict(ptp_amp=ptp_amp, std=std, max_acf=max_acf,
                dom_power_ratio=dom_power_ratio, slope=float(coef[0]))

def _class_from_features(feat, rng):
    # permissive thresholds to make bands visible
    A1, A2 = 0.015 * rng, 0.05 * rng   # amplitude gates
    R2, R3 = 0.30, 0.50                # ACF gates
    P2, P3 = 0.30, 0.50                # spectral gates
    if feat["ptp_amp"] < A1 and feat["std"] < 0.01: return "STEADY"
    if feat["ptp_amp"] >= A2 and feat["max_acf"] >= R3 and feat["dom_power_ratio"] >= P3: return "STRONG_WAVE"
    if feat["ptp_amp"] >= A1 and feat["max_acf"] >= R2 and feat["dom_power_ratio"] >= P2: return "MILD_WAVE"
    if abs(feat["slope"]) > 1e-3 and feat["std"] < 0.02: return "DRIFT"
    if feat["dom_power_ratio"] < 0.2 and feat["max_acf"] < 0.2: return "BURSTY_NOISY"
    return "UNCERTAIN"

def classify_windows_samples(od, win_samples=300, overlap=0.5):
    """
    Return list of (i0, i1, label) over the OD sequence.
    Window is in number of samples to avoid requiring uniform timestamps.
    """
    x = np.asarray(od, dtype=float)
    n = len(x)
    if n < max(16, win_samples): return []
    step = max(1, int(win_samples * (1.0 - overlap)))
    rng = float(np.nanmax(x) - np.nanmin(x)) if n else 0.0
    out = []
    for i0 in range(0, n - win_samples + 1, step):
        i1 = i0 + win_samples
        y = x[i0:i1]
        t = np.arange(len(y), dtype=float)   # samples as “time”
        feat = _win_features(y, t)
        lab = _class_from_features(feat, rng if rng > 0 else 1.0)
        out.append((i0, i1, lab))
    return out

# ========================= Data Store =========================
class DataStore:
    def __init__(self):
        self.path = None
        self.sheet = None
        self.ts = []          # list[str] (stringified timestamps for now)
        self.od = []          # list[float]
        self.last_loaded_rows = 0
        self.classes = []     # list of (start_idx, end_idx, label)

    def load_excel(self, path, sheet_name=None):
        """
        Load Excel and pick time/OD columns.
        If sheet_name is None: pick the sheet with most numeric cells.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        xls = pd.ExcelFile(path, engine="openpyxl")
        target_sheet = sheet_name
        if target_sheet is None:
            best, best_count = None, -1
            for sh in xls.sheet_names:
                try:
                    tmp = pd.read_excel(xls, sheet_name=sh)
                except Exception:
                    continue
                num = pd.to_numeric(tmp.select_dtypes(include=["number"]).stack(), errors="coerce")
                count = int(num.notna().sum())
                if count > best_count:
                    best, best_count = sh, count
            target_sheet = best or xls.sheet_names[0]

        df = pd.read_excel(xls, sheet_name=target_sheet)
        df.columns = [str(c) for c in df.columns]

        # infer likely time & OD columns
        tcol = pick(df.columns, DATA_COL_GUESSES["time"]) or df.columns[0]
        ycol = pick(df.columns, DATA_COL_GUESSES["od"])
        if ycol is None:
            # first numeric-ish that's not time
            for c in df.columns:
                if c == tcol: continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    ycol = c; break
        if ycol is None:
            ycol = df.columns[min(1, len(df.columns) - 1)]

        ts = pd.to_datetime(df[tcol], errors="coerce")
        od = pd.to_numeric(df[ycol], errors="coerce")
        keep = ts.notna() & od.notna()
        ts = ts[keep].astype(str).tolist()
        od = od[keep].astype(float).tolist()
        if not od:
            raise ValueError(f"No numeric OD values found in sheet '{target_sheet}'.")

        self.path = path
        self.sheet = target_sheet
        self.ts, self.od = ts, od
        self.last_loaded_rows = len(od)

        # Pre-compute bands for History + Results
        self.classes = classify_windows_samples(self.od, win_samples=300, overlap=0.5)

    # ---- metrics used elsewhere ----
    @staticmethod
    def _linreg_slope(y):
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
        slope = abs(self.trend_slope(n))
        p2p   = self.volatility_p2p(n)
        slope_norm = min(1.0, slope / (spec_band/200.0 + 1e-9))
        p2p_norm   = min(1.0, p2p   / (spec_band*2.0 + 1e-9))
        score = 100.0 * (0.6 * p2p_norm + 0.4 * slope_norm)
        return max(0.0, min(100.0, score))

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
        self._draw_static(); self.set_value(0)

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
            self.canvas.create_text(w//2, h//2, text="(Load Excel to see history)", fill="#6B7280")
            return

        ymin, ymax = min(y), max(y)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0

        full_n = len(DATA.od)
        win_n = len(y)
        start_idx = max(0, full_n - win_n)

        def X(i):  return pad + (w-2*pad) * (i/(len(y)-1))
        # colored bands (stipple to fake transparency)
        if getattr(DATA, "classes", None):
            for s, e, label in DATA.classes:
                a = max(s, start_idx); b = min(e, full_n)
                if b <= a: continue
                xa = max(0, min(win_n-1, a - start_idx))
                xb = max(0, min(win_n-1, b - start_idx))
                if xb <= xa: continue
                x0 = X(xa); x1 = X(xb)
                color = CLASS_COLORS.get(label, "#BBBBBB")
                self.canvas.create_rectangle(x0, pad, x1, h-pad, fill=color, outline="", stipple="gray25")

        def Y(v):  return h-pad - (h-2*pad) * ((v - ymin)/(ymax - ymin))
        for i in range(1, len(y)):
            self.canvas.create_line(X(i-1), Y(y[i-1]), X(i), Y(y[i]), fill="#93C5FD", width=1)

        k = max(5, len(y)//50)
        sm, s = [], 0.0
        for i, v in enumerate(y):
            s += v
            if i >= k: s -= y[i-k]
            sm.append(s / min(i+1, k))
        for i in range(1, len(sm)):
            self.canvas.create_line(X(i-1), Y(sm[i-1]), X(i), Y(sm[i]), fill="#2563EB", width=2)

        slope = DATA.trend_slope(min(1024, len(y)))
        color = "#DC2626" if slope > 0 else ("#16A34A" if slope < 0 else "#6B7280")
        label = "Uptrend" if slope > 0 else ("Downtrend" if slope < 0 else "Stable")
        self.canvas.create_text(w-pad-70, pad+14, text=label, fill=color, font=("Segoe UI", 10, "bold"))
        ax = w - pad - 30; ay = pad + 28
        dy = -16 if slope > 0 else (16 if slope < 0 else 0)
        self.canvas.create_line(ax-10, ay, ax+10, ay+dy, arrow=tk.LAST, width=3, fill=color)

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
        for i in range(10): controls.columnconfigure(i, weight=1)
        ttk.Button(controls, text="Load Excel…", command=self.load_excel).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Open in VS Code", command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=1, sticky="w")

        self.info = ttk.Label(self, text="No file loaded.", foreground="#6B7280")
        self.info.grid(row=2, column=0, sticky="w")

        table_area = ttk.Frame(self); table_area.grid(row=3, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(3, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

    def load_excel(self):
        path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx;*.xls;*.xlsm"), ("All files", "*.*")]
        )
        if not path: return
        try:
            DATA.load_excel(path)   # auto-picks best sheet
            base = os.path.basename(path)
            sheet = DATA.sheet or "(first)"
            self.info.config(text=f"Loaded: {base}  •  sheet={sheet}  •  rows={len(DATA.od)}")
            App.status("Excel loaded. History & Results now using real data.")
        except Exception as e:
            messagebox.showerror("Load Excel failed", str(e))
            App.status("Excel load failed")

class ResultsPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Results")

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

        # LEFT: real chart
        left = ttk.Frame(grid); left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,8))
        ttk.Label(left, text="OD vs Time (Live)", style="Subhead.TLabel").pack(anchor="w")
        self.fig_res = Figure(figsize=(6,4), dpi=100)
        self.ax_res = self.fig_res.add_subplot(111)
        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=left)
        self.canvas_res.draw()
        self.canvas_res.get_tk_widget().pack(fill="both", expand=True, pady=(6,0))
        self._last_drawn_len = -1

        # RIGHT: gauge + KPIs
        right_top = ttk.Frame(grid); right_top.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_top, text="OK / NG Indicator", style="Subhead.TLabel").pack(anchor="w")
        self.gauge = Gauge(right_top, width=360, height=200); self.gauge.pack(fill="both", expand=True, pady=(6,0))

        right_bot = ttk.Frame(grid); right_bot.grid(row=1, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_bot, text="Model Metrics", style="Subhead.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        right_bot.columnconfigure((0,1,2,3), weight=1); right_bot.rowconfigure(1, weight=1)

        mat = ttk.Frame(right_bot, padding=6, relief="ridge"); mat.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0,6))
        for i in range(3): mat.columnconfigure(i, weight=1); mat.rowconfigure(i, weight=1)
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

        self.after(500, self._tick)

    def _redraw_results_plot(self):
        self.ax_res.clear()
        y = DATA.recent_window(2000)  # last ~2000 samples
        self.ax_res.set_title("Latest OD with waviness bands")
        self.ax_res.set_xlabel("sample"); self.ax_res.set_ylabel("OD")

        if not y:
            self.ax_res.text(0.5, 0.5, "Load Excel to view plot", ha="center", va="center")
            self.canvas_res.draw(); return

        x = np.arange(len(y))
        y = np.asarray(y, dtype=float)

        # shaded bands
        bands = classify_windows_samples(y, win_samples=300, overlap=0.5)
        for i0, i1, lab in bands:
            self.ax_res.axvspan(i0, i1, color=CLASS_COLORS.get(lab, "#BBBBBB"), alpha=0.18, lw=0)

        # raw + smooth
        self.ax_res.plot(x, y, lw=1, alpha=0.55, label="raw")
        k = max(5, len(y)//120)
        if k > 1:
            sm = np.convolve(y, np.ones(k)/k, mode="same")
            self.ax_res.plot(x, sm, lw=2.5, label="smooth")
        self.ax_res.legend(loc="upper left")
        self.fig_res.tight_layout()
        self.canvas_res.draw()

    def predict_latest(self):
        tp, tn, fp, fn = [random.randint(20, 60) for _ in range(4)]
        acc = (tp + tn) / max(1, (tp+tn+fp+fn))
        prec = tp / max(1, (tp+fp)); rec = tp / max(1, (tp+fn))
        f1 = 2*prec*rec / max(1e-9, (prec+rec))
        self.cm_tp.config(text=str(tp)); self.cm_tn.config(text=str(tn))
        self.cm_fp.config(text=str(fp)); self.cm_fn.config(text=str(fn))
        self.k_acc.config(text=f"Accuracy: {acc*100:0.1f}%")
        self.k_prec.config(text=f"Precision: {prec*100:0.1f}%")
        self.k_rec.config(text=f"Recall: {rec*100:0.1f}%")
        self.k_f1.config(text=f"F1 Score: {f1*100:0.1f}%")
        App.busy("Updated metrics from demo numbers")

    def _tick(self):
        # update gauge
        if DATA.od:
            pct = DATA.ng_score(n=1024, spec_mm=10.0, spec_band=0.02)
        else:
            pct = 50
        self.gauge.set_value(pct)

        # refresh left plot when new data appears
        cur_len = len(DATA.od)
        if cur_len != self._last_drawn_len:
            self._redraw_results_plot()
            self._last_drawn_len = cur_len

        self.after(1000, self._tick)

class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Trend")
        desc = "Trend view shows last N samples with smoothed curve and up/down indicator.\n" \
               "Colored background bands mark waviness classes."
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

# --- (Optional) Analysis tab reused as-is; kept for completeness ---
class FeatureEngine:
    """
    Builds windowed features from OD (numpy array).
    Produces the 6-panel figure and supports k-fold curves.
    """
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
            y = w.astype(float)
            mean = float(np.mean(y))
            std  = float(np.std(y))
            p2p  = float(np.max(y) - np.min(y))
            mad  = float(np.mean(np.abs(y - np.median(y))))
            rel_max = (np.max(y) - mean) / max(1e-9, mean)
            rel_min = (mean - np.min(y)) / max(1e-9, mean)
            coef_var = std / max(1e-9, mean)
            norm_var = (std**2) / max(1e-9, mean**2)
            ptp_ratio = p2p / max(1e-9, std)
            rows.append({
                "mean_abs_diff": mad,
                "coef_variation": coef_var,
                "relative_range": p2p / max(1e-9, mean),
                "normalized_variance": norm_var,
                "peak_to_peak_ratio": ptp_ratio,
                "relative_max_deviation": rel_max,
                "relative_min_deviation": rel_min,
                "max_abs_diff": p2p,
            })
        df = pd.DataFrame(rows)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def demo_labels(self, df):
        thr = df["mean_abs_diff"].median() + df["mean_abs_diff"].std()
        return (df["mean_abs_diff"] > thr).astype(int)  # 1 = bad, 0 = good

    # ✅ Signature matches fig_sixpack(df.values, y, clf)
    def kfold_curve(self, X, y, model, windows=[25, 50, 75, 100, 125]):
        """
        Compute average accuracy across window sizes using 5-fold CV.
        X is unused (kept for signature compatibility).
        """
        sizes, accs = [], []
        for w in windows:
            self.win = max(8, int(w * self.fs))
            self.step = max(1, self.win // 2)
            dfw = self.feature_table()
            if len(dfw) < 30:
                continue
            Xw = dfw.values
            yw = self.demo_labels(dfw) if y is None else pd.Series(y).reindex(range(len(dfw))).fillna(method="ffill").fillna(method="bfill").astype(int).values
            # guard against class imbalance / too few points
            if len(np.unique(yw)) < 2:
                continue
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            scaler = StandardScaler()
            for tr, te in skf.split(Xw, yw):
                Xtr, Xte = Xw[tr], Xw[te]
                ytr, yte = yw[tr], yw[te]
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)
                model.fit(Xtr, ytr)
                scores.append(model.score(Xte, yte))
            sizes.append(w)
            accs.append(float(np.mean(scores)) if scores else np.nan)
        return np.array(sizes), np.array(accs)

    def fig_sixpack(self):
        df = self.feature_table()
        if df.empty:
            fig = Figure(figsize=(11.5, 6.6), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Not enough data for analysis.\nLoad a longer Excel.", ha="center", va="center")
            ax.axis("off")
            return fig

        y = self.demo_labels(df)

        # Use constrained layout to prevent overlap in Tk
        fig = Figure(figsize=(11.5, 6.6), dpi=100, constrained_layout=True)
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            pass

        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0])

        # 1) Accuracy vs window
        ax1 = fig.add_subplot(gs[0, 0])
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        sizes, acc_rf = self.kfold_curve(df.values, y, clf)
        ax1.plot(sizes, np.minimum(1.0, acc_rf*0.998), marker="o", label="Random Forest", lw=2)
        ax1.plot(sizes, np.minimum(1.0, acc_rf*0.992), marker="o", label="Logistic Regression (demo)", lw=1.3)
        ax1.plot(sizes, np.minimum(1.0, acc_rf*0.989), marker="o", label="SVM (demo)", lw=1.3)
        ax1.set_title("Average model accuracy vs. window size\n5-fold CV", fontsize=10)
        ax1.set_xlabel("Window size (sec)", fontsize=9); ax1.set_ylabel("Average accuracy", fontsize=9)
        ax1.tick_params(labelsize=8); ax1.legend(fontsize=8, loc="lower left")

        # 2) Feature importance
        ax2 = fig.add_subplot(gs[0, 1])
        scaler = StandardScaler().fit(df.values)
        Xs = scaler.transform(df.values)
        clf.fit(Xs, y)
        importances = clf.feature_importances_
        order = np.argsort(importances)
        ax2.barh(np.array(df.columns)[order], importances[order])
        ax2.set_title("Random Forest feature importance", fontsize=10)
        ax2.set_xlabel("Feature weight", fontsize=9)
        ax2.tick_params(axis="y", labelsize=8); ax2.tick_params(axis="x", labelsize=8)

        # 3) MAD histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(df.loc[y==0, "mean_abs_diff"], bins=40, alpha=0.7, label="Good")
        ax3.hist(df.loc[y==1, "mean_abs_diff"], bins=40, alpha=0.7, label="Rejected, chatter")
        ax3.set_title("Mean Absolute Difference (smoothness)", fontsize=10)
        ax3.set_xlabel("MAD", fontsize=9); ax3.set_ylabel("Count", fontsize=9)
        ax3.tick_params(labelsize=8); ax3.legend(fontsize=8, loc="upper right")

        # 4) PCA scatter
        ax4 = fig.add_subplot(gs[1, 0])
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        ax4.scatter(Xp[y==0,0], Xp[y==0,1], s=8, alpha=0.5, label="Good")
        ax4.scatter(Xp[y==1,0], Xp[y==1,1], s=10, alpha=0.8, label="Bad (Chatter)", color="red")
        ax4.set_title("Principal component analysis", fontsize=10)
        ax4.set_xlabel(f"PC1: {pca.explained_variance_ratio_[0]*100:0.1f}% var", fontsize=9)
        ax4.set_ylabel(f"PC2: {pca.explained_variance_ratio_[1]*100:0.1f}% var", fontsize=9)
        ax4.tick_params(labelsize=8); ax4.legend(fontsize=8, loc="best")

        # 5) PCA loadings
        ax5 = fig.add_subplot(gs[1, 1])
        comps = pca.components_
        ax5.axvline(0, color="0.7", linewidth=0.8); ax5.axhline(0, color="0.7", linewidth=0.8)
        for i, name in enumerate(df.columns):
            ax5.arrow(0, 0, comps[0,i], comps[1,i], head_width=0.02, head_length=0.02,
                      fc='k', ec='k', alpha=0.55, length_includes_head=True)
            ax5.text(comps[0,i]*1.08, comps[1,i]*1.08, name, fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc="#FFFBEA", ec="#CFCFCF"))
        ax5.set_xlim(-1.0, 1.0); ax5.set_ylim(-1.0, 1.0)
        ax5.set_title("Feature weights on PCs", fontsize=10)
        ax5.set_xlabel("PC1 weights", fontsize=9); ax5.set_ylabel("PC2 weights", fontsize=9)
        ax5.tick_params(labelsize=8)

        # 6) Correlation heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        corr = df.corr().values
        im = ax6.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax6.set_xticks(range(len(df.columns))); ax6.set_xticklabels(df.columns, rotation=50, ha="right", fontsize=8)
        ax6.set_yticks(range(len(df.columns))); ax6.set_yticklabels(df.columns, fontsize=8)
        ax6.set_title("Feature correlation matrix", fontsize=10)
        cbar = fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8); cbar.set_label("Correlation", fontsize=9)

        return fig



class AnalysisPage(BasePage):
    """Six-pack of modeling/feature plots (Compute from current data)."""
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Modeling & Analysis")

        # Top controls
        top = ttk.Frame(self); top.grid(row=1, column=0, sticky="ew", pady=(0,8))
        ttk.Button(top, text="Compute from current data", command=self.render).grid(row=0, column=0, sticky="w")
        self.status_lbl = ttk.Label(top, text="", foreground="#6B7280")
        self.status_lbl.grid(row=0, column=1, sticky="w", padx=12)

        # Canvas holder
        self.fig = None
        self.canvas_widget = None
        area = ttk.Frame(self); area.grid(row=2, column=0, sticky="nsew")
        self.rowconfigure(2, weight=1); self.columnconfigure(0, weight=1)
        self.canvas_container = area

    def render(self):
        # Make sure we have enough samples to build several windows
        if not DATA.od or len(DATA.od) < 400:
            self.status_lbl.config(text="Load an Excel with a few hundred OD samples first.")
            return

        # Build the figure using FeatureEngine (ensure your FeatureEngine class is in the file)
        fe = FeatureEngine(np.array(DATA.od, dtype=float), window=30, fs=1.0)
        fig = fe.fig_sixpack()

        # Replace existing canvas
        if self.canvas_widget:
            try:
                self.canvas_widget.get_tk_widget().destroy()
            except Exception:
                pass

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
        filemenu.add_command(label="Open Project in VS Code", command=lambda: open_in_vscode(os.path.abspath(".")))
        filemenu.add_separator(); filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_sidebar(self, parent):
        bar = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        ttk.Label(bar, text="Wavy Detection", foreground="white", background="#111827",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0,16))
        for name, accel in [("Data","Ctrl+1"),("Results","Ctrl+2"),("History","Ctrl+3"),("Analysis","Ctrl+4")]:
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

if __name__ == "__main__":
    App().mainloop()
