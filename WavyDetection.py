# te_qc_gui.py
# Wavy Detection — Prototype Dashboard with History/Trend + Data-driven Gauge
# Built only with Python stdlib.

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
APP_VERSION = "v0.3 (history + data needle)"

DATA_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime"],
    "od":   ["od", "outer_diameter", "tube_od", "ndc_od_value"],
}
def rolling_mad(x):
    m = np.median(x)
    return np.mean(np.abs(x - m))

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
    def __init__(self):
        self.path = None
        self.ts = []   # list[datetime or str]
        self.od = []   # list[float]
        self.last_loaded_rows = 0

    def load_csv(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
            sn = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            dialect = sn.sniff(sample) if sample else csv.excel
            rdr = csv.DictReader(f, dialect=dialect)
            cols = rdr.fieldnames or []
            # pick columns
            tcol = pick(cols, DATA_COL_GUESSES["time"]) or cols[0]
            ycol = pick(cols, DATA_COL_GUESSES["od"])   or (cols[1] if len(cols) > 1 else cols[0])
            ts, od = [], []
            for row in rdr:
                y = try_float(row.get(ycol, ""))
                if y is None: continue
                od.append(y)
                ts.append(row.get(tcol) or "")
        if not od:
            raise ValueError("No numeric OD values found in CSV.")
        self.path = path
        self.ts, self.od = ts, od
        self.last_loaded_rows = len(od)

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
        slope_norm = min(1.0, slope / (spec_band/200.0 + 1e-9))  # arbitrary slope scale
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

class FeatureEngine:
    """
    Builds windowed features from DATA (uses only OD to start).
    Produces the 6-panel figure similar to your screenshots.
    """
    def __init__(self, od, window=30, fs=1.0):
        # od: numpy array of OD, window: seconds, fs: samples/sec
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
            # “relative” max/min devs vs mean
            rel_max = (np.max(y) - mean) / max(1e-9, mean)
            rel_min = (mean - np.min(y)) / max(1e-9, mean)
            coef_var = std / max(1e-9, mean)
            # normalized variance (unitless)
            norm_var = (std**2) / max(1e-9, mean**2)
            # peak-to-peak ratio = p2p / std
            ptp_ratio = p2p / max(1e-9, std)

            rows.append({
                "mean_abs_diff": mad,
                "coef_variation": coef_var,
                "relative_range": rel_range,
                "normalized_variance": norm_var,
                "peak_to_peak_ratio": ptp_ratio,
                "relative_max_deviation": rel_max,
                "relative_min_deviation": rel_min,
                "max_abs_diff": p2p,  # used in your heatmap
            })
        df = pd.DataFrame(rows)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    # ---- demo labels (Good vs Bad) purely from MAD threshold so plots render now
    def demo_labels(self, df):
        thr = df["mean_abs_diff"].median() + df["mean_abs_diff"].std()
        y = (df["mean_abs_diff"] > thr).astype(int)  # 1=Bad/chatter, 0=Good
        return y

    def kfold_curve(self, X, y, model, windows=[10,20,30,40,60,90,120]):
        """Return window size vs accuracy (5-fold)."""
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
                ytr, yte = yw.iloc[tr] if isinstance(yw, pd.Series) else yw[tr], yw.iloc[te] if isinstance(yw, pd.Series) else yw[te]
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

        # labels (demo)
        y = self.demo_labels(df)

        # 1) Accuracy curve (LogReg, RF, SVM-like using RF twice for demo)
        fig = Figure(figsize=(12,6), dpi=100)
        gs = fig.add_gridspec(2,3, wspace=0.35, hspace=0.35)

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        sizes, acc_rf = self.kfold_curve(df.values, y, clf)
        ax1 = fig.add_subplot(gs[0,0])
        for mult, lab in [(0.98,"Logistic Regression (demo)"), (1.00,"Random Forest"), (0.995,"SVM (demo)")]:
            ax1.plot(sizes, np.minimum(1.0, acc_rf*mult), marker="o", label=lab)
        ax1.set_title("Average model accuracy vs. Window size\n5-split K-Folds")
        ax1.set_xlabel("Window size (sec)"); ax1.set_ylabel("Average accuracy"); ax1.legend()

        # 2) RF feature importance
        scaler = StandardScaler().fit(df.values)
        Xs = scaler.transform(df.values)
        clf.fit(Xs, y)
        importances = clf.feature_importances_
        order = np.argsort(importances)
        ax2 = fig.add_subplot(gs[0,1])
        ax2.barh(np.array(df.columns)[order], importances[order])
        ax2.set_title("Random Forest feature importance\nWindow size: ~30 sec")
        ax2.set_xlabel("Feature weight")

        # 3) MAD histogram (good vs bad)
        ax3 = fig.add_subplot(gs[0,2])
        ax3.hist(df.loc[y==0,"mean_abs_diff"], bins=40, alpha=0.6, label="Good")
        ax3.hist(df.loc[y==1,"mean_abs_diff"], bins=40, alpha=0.6, label="Rejected, chatter")
        ax3.set_title("Mean Absolute Difference (smoothness indicator)\nWindow size: ~30 sec")
        ax3.set_xlabel("Mean Absolute Difference"); ax3.set_ylabel("Density"); ax3.legend()

        # 4) PCA scatter (PC1 vs PC2)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        ax4 = fig.add_subplot(gs[1,0])
        ax4.scatter(Xp[y==0,0], Xp[y==0,1], s=8, alpha=0.5, label="Good")
        ax4.scatter(Xp[y==1,0], Xp[y==1,1], s=8, alpha=0.8, label="Bad (Chatter)", color="red")
        ax4.set_title("Principal component analysis")
        ax4.set_xlabel(f"PC1: {pca.explained_variance_ratio_[0]*100:0.1f}% variance")
        ax4.set_ylabel(f"PC2: {pca.explained_variance_ratio_[1]*100:0.1f}% variance")
        ax4.legend(loc="best")

        # 5) PCA loadings (feature weights on PCs)
        ax5 = fig.add_subplot(gs[1,1])
        comps = pca.components_
        ax5.axvline(0,color="gray",linewidth=0.8); ax5.axhline(0,color="gray",linewidth=0.8)
        for i, name in enumerate(df.columns):
            ax5.arrow(0, 0, comps[0,i], comps[1,i], head_width=0.02, head_length=0.02, fc='k', ec='k', alpha=0.6)
            ax5.text(comps[0,i]*1.08, comps[1,i]*1.08, name, fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="#FFF9C4", ec="#999"))
        ax5.set_xlim(-1,1); ax5.set_ylim(-1,1)
        ax5.set_title("Feature weights on principal components")
        ax5.set_xlabel("PC1 weights"); ax5.set_ylabel("PC2 weights")

        # 6) Correlation heatmap
        ax6 = fig.add_subplot(gs[1,2])
        corr = df.corr().values
        im = ax6.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax6.set_xticks(range(len(df.columns))); ax6.set_xticklabels(df.columns, rotation=60, ha="right", fontsize=8)
        ax6.set_yticks(range(len(df.columns))); ax6.set_yticklabels(df.columns, fontsize=8)
        ax6.set_title("Feature Correlation Matrix")
        cbar = fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation")
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
        for i in range(10): controls.columnconfigure(i, weight=1)
        ttk.Button(controls, text="Load CSV…", command=self.load_csv).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Import Folder…", command=lambda: App.busy("Import folder (todo)")).grid(row=0, column=1, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Run Preprocess", command=lambda: App.busy("Preprocess (todo)")).grid(row=0, column=2, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Compute FFT", command=lambda: App.busy("FFT (todo)")).grid(row=0, column=3, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Open in VS Code", command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=4, sticky="w")

        self.info = ttk.Label(self, text="No file loaded.", foreground="#6B7280")
        self.info.grid(row=2, column=0, sticky="w")

        table_area = ttk.Frame(self); table_area.grid(row=3, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(3, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files","*.*")]
        )
        if not path: return
        try:
            DATA.load_csv(path)
            self.info.config(text=f"Loaded: {os.path.basename(path)}  •  rows={len(DATA.od)}")
            App.status("CSV loaded. History & gauge now using real data.")
        except Exception as e:
            messagebox.showerror("Load CSV failed", str(e))
            App.status("CSV load failed")

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
        self.left_placeholder = ttk.Label(left, style="Placeholder.TLabel", relief="ridge", padding=24,
                                          text="(Time-series plot placeholder)\nSolid: measured OD\nDashed: forecast\nToggles: pressure, temp, speed")
        self.left_placeholder.pack(fill="both", expand=True, pady=(6,0))

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

        self.after(1000, self._tick)

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
        if DATA.od:
            pct = DATA.ng_score(n=1024, spec_mm=10.0, spec_band=0.02)
        else:
            pct = 50 + 15 * math.sin(datetime.now().timestamp()/2.0)
        self.gauge.set_value(pct)
        self.after(1000, self._tick)

class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Trend")
        desc = "Trend view shows last N samples with smoothed curve and up/down indicator.\n" \
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
    """Simple test area so the pages dict resolves."""
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
    """Six-pack of modeling/feature plots (like your mockups)."""
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
        self.canvas_container = area  # placeholder parent

    def render(self):
        if not DATA.od or len(DATA.od) < 400:
            self.status_lbl.config(text="Load a CSV with at least a few hundred samples.")
            return
        fe = FeatureEngine(np.array(DATA.od), window=30, fs=1.0)
        fig = fe.fig_sixpack()

        # embed / replace canvas
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
