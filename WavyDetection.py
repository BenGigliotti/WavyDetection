# Wavy Detection Prototype Dashboard
# By Ben Gigliotti, Anubhav Sen, Ariel Michalowski, Rayleen Fu, Francisco Verdugo
# Did some digging we can easily implement AI into our GUI we can just use numpy for our FFT and pandas for our CSV handling ez pz.

import os, subprocess, math, random
import tkinter as tk
from tkinter import ttk
from datetime import datetime

APP_TITLE   = "Wavy Detection Prototype Dashboard"
APP_VERSION = "v0.2 (blueprint layout)"

# Our utilities
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

#Simple Gauge Widget
class Gauge(ttk.Frame):
    """
    Semicircle gauge: 0% (left, OK) → 100% (right, NG)
    set_value(pct: 0..100) updates needle & label
    """
    def __init__(self, parent, width=360, height=200, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.width, self.height = width, height
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        self._needle = None
        self._pct_label = ttk.Label(self, text="— %", font=("Segoe UI", 12, "bold"))
        self._pct_label.grid(row=1, column=0, pady=(8,0))
        self._status = ttk.Label(self, text="Status: ", font=("Segoe UI", 11))
        self._status.grid(row=2, column=0)
        self._draw_static()
        self.set_value(0)

    def _draw_static(self):
        w, h = self.width, self.height
        cx, cy, r = w//2, h-10, min(w, h*2)//2 - 10
        #this just creates green (OK) → yellow → red (NG)
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=60,
                               fill="#16A34A", outline="")
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=240, extent=60,
                               fill="#D97706", outline="")
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=300, extent=60,
                               fill="#DC2626", outline="")
        #for each ticks
        for i in range(0, 11):
            ang = math.radians(180 + i*18)
            x0 = cx + (r-18)*math.cos(ang); y0 = cy + (r-18)*math.sin(ang)
            x1 = cx + (r-2)*math.cos(ang);  y1 = cy + (r-2)*math.sin(ang)
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill="#334155")
        #our labels
        self.canvas.create_text(cx - r + 40, cy - 20, text="OK", font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(cx + r - 40, cy - 20, text="NG", font=("Segoe UI", 11, "bold"))

        self._cx, self._cy, self._r = cx, cy, r

    def set_value(self, pct: float):
        pct = max(0.0, min(100.0, float(pct)))
        #map 0..100 → 180°..360° just for the semicircle really
        ang = math.radians(180 + 180 * pct / 100.0)
        cx, cy, r = self._cx, self._cy, self._r
        x = cx + (r-26) * math.cos(ang)
        y = cy + (r-26) * math.sin(ang)
        #just to delete the old needle
        if self._needle: self.canvas.delete(self._needle)
        self._needle = self.canvas.create_line(cx, cy, x, y, width=5, fill="#111827", capstyle=tk.ROUND)
        #the text
        self._pct_label.config(text=f"{pct:0.1f}% confidence")
        status = "NG" if pct >= 50 else "OK"
        self._status.config(text=f"Status: {status}",
                            foreground=("#DC2626" if status=="NG" else "#16A34A"))

#base page
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

#this will start all of our pages
#we can edit these as we see fit
class DataPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Data")
        controls = ttk.Frame(self); controls.grid(row=1, column=0, sticky="ew", pady=(0,12))
        for i in range(8): controls.columnconfigure(i, weight=1)
        ttk.Button(controls, text="Load CSV…", command=lambda: App.busy("Select CSV")).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Import Folder…", command=lambda: App.busy("Import folder")).grid(row=0, column=1, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Run Preprocess", command=lambda: App.busy("Preprocess running")).grid(row=0, column=2, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Compute FFT", command=lambda: App.busy("FFT computing")).grid(row=0, column=3, sticky="w", padx=(0,8))
        ttk.Button(controls, text="Open in VS Code", command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=4, sticky="w")
        #data preview placeholder (we can replace this with our real table later)
        table_area = ttk.Frame(self); table_area.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

class ResultsPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Results (Blueprint)")

        #top controls
        tools = ttk.Frame(self); tools.grid(row=1, column=0, sticky="ew", pady=(0,12))
        ttk.Button(tools, text="Predict Latest", command=self.predict_latest).grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Button(tools, text="Export Report", command=lambda: App.busy("Export report")).grid(row=0, column=1, sticky="w", padx=(0,8))
        ttk.Button(tools, text="Open in VS Code", command=lambda: open_in_vscode(os.path.abspath("."))).grid(row=0, column=2, sticky="w")

        #3-sub-layout grid
        grid = ttk.Frame(self); grid.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        grid.columnconfigure(0, weight=2)  #big left panel
        grid.columnconfigure(1, weight=1)  #right column
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        #left big panel our time-series placeholder
        left = ttk.Frame(grid); left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,8))
        ttk.Label(left, text="OD vs Time (Live)", style="Subhead.TLabel").pack(anchor="w")
        self.left_placeholder = ttk.Label(left, style="Placeholder.TLabel", relief="ridge", padding=24,
                                          text="(Time-series plot placeholder)\nSolid: measured OD\nDashed: model forecast\nToggles: pressure, temp, speed")
        self.left_placeholder.pack(fill="both", expand=True, pady=(6,0))

        #right top the gauge
        right_top = ttk.Frame(grid); right_top.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_top, text="OK / NG Indicator", style="Subhead.TLabel").pack(anchor="w")
        self.gauge = Gauge(right_top, width=360, height=200); self.gauge.pack(fill="both", expand=True, pady=(6,0))

        #right bottom the metrics (confusion + KPIs)
        right_bot = ttk.Frame(grid); right_bot.grid(row=1, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_bot, text="Model Metrics", style="Subhead.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        right_bot.columnconfigure((0,1,2,3), weight=1); right_bot.rowconfigure(1, weight=1)

        #confusion matrix (2x2)
        mat = ttk.Frame(right_bot, padding=6, relief="ridge"); mat.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0,6))
        for i in range(3): mat.columnconfigure(i, weight=1)
        for i in range(3): mat.rowconfigure(i, weight=1)
        ttk.Label(mat, text="Predicted", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, columnspan=2)
        ttk.Label(mat, text="Actual", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, rowspan=2, sticky="s")
        ttk.Label(mat, text="OK").grid(row=1, column=1);   ttk.Label(mat, text="NG").grid(row=1, column=2)
        ttk.Label(mat, text="OK").grid(row=2, column=0, sticky="e")
        self.cm_tn = ttk.Label(mat, text="Nothing", style="KPI.TLabel"); self.cm_tn.grid(row=2, column=1)
        self.cm_fp = ttk.Label(mat, text="Nothing", style="KPI.TLabel"); self.cm_fp.grid(row=2, column=2)
        ttk.Label(mat, text="NG").grid(row=3, column=0, sticky="e")
        self.cm_fn = ttk.Label(mat, text="Nothing", style="KPI.TLabel"); self.cm_fn.grid(row=3, column=1)
        self.cm_tp = ttk.Label(mat, text="Nothing", style="KPI.TLabel"); self.cm_tp.grid(row=3, column=2)

        #KPI
        kpis = ttk.Frame(right_bot); kpis.grid(row=1, column=2, columnspan=2, sticky="nsew")
        for i in range(2): kpis.columnconfigure(i, weight=1)
        self.k_acc = ttk.Label(kpis, text="Accuracy: Nothing here yet! ", style="KPI.TLabel"); self.k_acc.grid(row=0, column=0, sticky="w", pady=(0,6))
        self.k_prec= ttk.Label(kpis, text="Precision: Nothing here yet! ", style="KPI.TLabel"); self.k_prec.grid(row=1, column=0, sticky="w", pady=(0,6))
        self.k_rec = ttk.Label(kpis, text="Recall: Nothing here yet! ",    style="KPI.TLabel"); self.k_rec.grid(row=0, column=1, sticky="w", pady=(0,6))
        self.k_f1  = ttk.Label(kpis, text="F1 Score: Nothing here yet! ",  style="KPI.TLabel"); self.k_f1.grid(row=1, column=1, sticky="w", pady=(0,6))

        #this is just to animate the gauge just a tick
        self.after(800, self._demo_tick)

    def predict_latest(self):
        App.busy("Predicting latest window…")
        #updating left placeholder here
        tp, tn, fp, fn = [random.randint(10, 40) for _ in range(4)]
        acc = (tp + tn) / max(1, (tp+tn+fp+fn))
        prec = tp / max(1, (tp+fp)); rec = tp / max(1, (tp+fn))
        f1 = 2*prec*rec / max(1e-9, (prec+rec))
        self.cm_tp.config(text=str(tp)); self.cm_tn.config(text=str(tn))
        self.cm_fp.config(text=str(fp)); self.cm_fn.config(text=str(fn))
        self.k_acc.config(text=f"Accuracy: {acc*100:0.1f}%")
        self.k_prec.config(text=f"Precision: {prec*100:0.1f}%")
        self.k_rec.config(text=f"Recall: {rec*100:0.1f}%")
        self.k_f1.config(text=f"F1 Score: {f1*100:0.1f}%")

    def _demo_tick(self):
        #20–80% just to show movement (this doesnt do anything XD)
        pct = 50 + 30 * math.sin(datetime.now().timestamp()/2.0)
        self.gauge.set_value(pct)
        self.after(800, self._demo_tick)

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

#main
class App(tk.Tk):
    _status_var = None

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x760"); self.minsize(1080, 680)

        self._init_style(); self._init_menu()

        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0); root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        sidebar = self._build_sidebar(root); sidebar.grid(row=0, column=0, sticky="nsw")
        self.container = ttk.Frame(root, padding=(12,16,16,16)); self.container.grid(row=0, column=1, sticky="nsew")
        self.container.columnconfigure(0, weight=1); self.container.rowconfigure(0, weight=1)

        self.pages = {"Data": DataPage(self.container), "Results": ResultsPage(self.container), "Tests": TestsPage(self.container)}
        for p in self.pages.values(): p.grid(row=0, column=0, sticky="nsew")
        self.show("Results")  # jump straight to the blueprint page for demos

        self._build_statusbar()
        self.bind("<Control-1>", lambda e: self.show("Data"))
        self.bind("<Control-2>", lambda e: self.show("Results"))
        self.bind("<Control-3>", lambda e: self.show("Tests"))

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
        for name, accel in [("Data","Ctrl+1"),("Results","Ctrl+2"),("Tests","Ctrl+3")]:
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
