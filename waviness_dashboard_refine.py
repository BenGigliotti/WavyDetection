# waviness_dashboard.py
# Visualize Ovality from NDC.xlsx (sheet NDC_System_Ovality_Value) with waviness highlights
# Run: streamlit run waviness_dashboard.py

from datetime import timedelta
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Colors & constants ----------------
CLASS_COLORS = {
    "STEADY": "#4CAF50",
    "MILD_WAVE": "#FFB300",
    "STRONG_WAVE": "#E53935",
    "DRIFT": "#2196F3",
    "BURSTY_NOISY": "#8E24AA",
    "UNCERTAIN": "#9E9E9E",
}
TIME_CANDIDATES = ["timestamp", "t_stamp", "time", "date_time", "datetime", "ts"]

# ---------------- Helpers ----------------
def normalize_label(s: str) -> str:
    if not isinstance(s, str):
        return "UNCERTAIN"
    t = s.strip().upper().replace("-", "_").replace(" ", "_").replace("/", "_")
    alias = {
        "BURSTY_NOISE": "BURSTY_NOISY",
        "NOISY": "BURSTY_NOISY",
        "MILD": "MILD_WAVE",
        "MILDWAVE": "MILD_WAVE",
        "STRONG": "STRONG_WAVE",
        "STRONGWAVE": "STRONG_WAVE",
    }
    return alias.get(t, t)

def pick_time_col(cols):
    low = [str(c).lower() for c in cols]
    for cand in TIME_CANDIDATES:
        if cand in low:
            return cols[low.index(cand)]
    return cols[0]  # fallback

# -------- Feature extraction --------
def _features(y, t):
    # y, t are numpy arrays; t in seconds relative to window start
    if len(y) < 3:
        return dict(ptp_amp=0.0, std=0.0, max_acf=0.0, dom_power_ratio=0.0, slope=0.0)
    # robust slope (polyfit in time)
    coef = np.polyfit(t, y, 1)
    trend = np.polyval(coef, t)
    yd = y - trend

    ptp_amp = float(np.nanmax(y) - np.nanmin(y))
    std = float(np.nanstd(y))

    # normalized ACF peak (excluding lag 0)
    z = yd - np.nanmean(yd)
    if np.allclose(z, 0):
        max_acf = 0.0
    else:
        acf = np.correlate(z, z, mode="full")
        acf = acf[len(z) - 1 :]
        if acf[0] != 0:
            acf = acf / acf[0]
        max_acf = float(np.nanmax(acf[1 : max(2, len(z) // 2)])) if len(z) > 2 else 0.0

    # dominant spectral power ratio
    Y = np.abs(np.fft.rfft(yd - np.nanmean(yd)))
    dom_power_ratio = float(Y[1:].max() / Y.sum()) if Y.sum() > 0 and len(Y) > 1 else 0.0

    return dict(
        ptp_amp=ptp_amp,
        std=std,
        max_acf=max_acf,
        dom_power_ratio=dom_power_ratio,
        slope=float(coef[0]),
    )

def _classify(feat, rng):
    # thresholds tuned to be a bit permissive so you see bands
    A1, A2 = 0.015 * rng, 0.05 * rng
    R2, R3 = 0.30, 0.50
    P2, P3 = 0.30, 0.50
    if feat["ptp_amp"] < A1 and feat["std"] < 0.01:
        return "STEADY"
    if feat["ptp_amp"] >= A2 and feat["max_acf"] >= R3 and feat["dom_power_ratio"] >= P3:
        return "STRONG_WAVE"
    if feat["ptp_amp"] >= A1 and feat["max_acf"] >= R2 and feat["dom_power_ratio"] >= P2:
        return "MILD_WAVE"
    if abs(feat["slope"]) > 1e-3 and feat["std"] < 0.02:
        return "DRIFT"
    if feat["dom_power_ratio"] < 0.2 and feat["max_acf"] < 0.2:
        return "BURSTY_NOISY"
    return "UNCERTAIN"

def compute_classes_time_windows(series: pd.Series, win_min=5, overlap=0.5) -> pd.DataFrame:
    """
    Time-based sliding windows.
    - No assumption of fixed sampling rate.
    - Windows advance by (1 - overlap) * window_size.
    """
    if series.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    s = series.sort_index().astype(float)
    rng = float(np.nanmax(s.values) - np.nanmin(s.values))
    if not np.isfinite(rng) or rng == 0:
        return pd.DataFrame(columns=["start", "end", "label"])

    win = pd.Timedelta(minutes=float(win_min))
    step = pd.Timedelta(minutes=float(win_min) * (1.0 - float(overlap)))
    if step <= pd.Timedelta(seconds=1):
        step = pd.Timedelta(seconds=1)

    t0 = s.index.min()
    t1 = s.index.max()
    rows = []
    cur = t0

    while cur + win <= t1:
        seg = s.loc[cur : cur + win]
        # require at least a few points
        if len(seg) >= 8 and np.isfinite(seg.values).sum() >= 8:
            # build time axis in seconds relative to segment start
            tt = (seg.index.view("int64") - seg.index.view("int64")[0]) / 1e9
            feat = _features(seg.values.astype(float), tt.astype(float))
            label = _classify(feat, rng)
            rows.append({"start": cur, "end": cur + win, "label": label, **feat})
        cur += step

    return pd.DataFrame(rows)

# -------- Data loader (Excel) --------
@st.cache_data(show_spinner=False)
def load_ovality_from_excel(xlsx_path: str, sheet="NDC_System_Ovality_Value"):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
    raw = pd.read_excel(xlsx_path, sheet_name=sheet)
    raw.columns = [str(c) for c in raw.columns]

    time_col = pick_time_col(raw.columns)
    if "Tag_value" in raw.columns:
        val_col = "Tag_value"
    else:
        # first numeric-ish column that's not time
        val_col = None
        for c in raw.columns:
            if c == time_col:
                continue
            s = pd.to_numeric(raw[c], errors="coerce")
            if s.notna().any():
                val_col = c
                break
        if val_col is None:
            val_col = raw.columns[min(1, len(raw.columns) - 1)]

    df = raw[[time_col, val_col]].copy()
    df.rename(columns={time_col: "timestamp", val_col: "Ovality"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["Ovality"] = pd.to_numeric(df["Ovality"], errors="coerce")
    df = df.dropna(subset=["Ovality"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

# ---------------- UI ----------------
st.set_page_config(page_title="Waviness Dashboard – Ovality (NDC.xlsx)", layout="wide")
st.title("Waviness Dashboard — NDC Ovality")

excel_path = st.sidebar.text_input("Excel file", value="NDC.xlsx")
sheet_name = "NDC_System_Ovality_Value"
st.sidebar.write(f"Sheet: **{sheet_name}** (fixed)")

try:
    ts = load_ovality_from_excel(excel_path, sheet=sheet_name)
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

# optional external classes CSV (if you have one)
classes_csv = st.sidebar.text_input("Classes CSV (optional)", value="")
classes_ext = pd.DataFrame()
if classes_csv.strip() and os.path.exists(classes_csv.strip()):
    # simple loader (start, end, label columns)
    c = pd.read_csv(classes_csv.strip())
    for col in c.columns:
        if str(col).lower() in ["start", "window_start", "timestamp"]:
            c["start"] = pd.to_datetime(c[col], errors="coerce")
        if str(col).lower() in ["end", "window_end", "time_end", "stop"]:
            c["end"] = pd.to_datetime(c[col], errors="coerce")
        if str(col).lower() in ["label", "class", "wave_class"]:
            c["label"] = c[col].astype(str).map(normalize_label)
    if {"start", "end", "label"} <= set(c.columns):
        classes_ext = c.dropna(subset=["start", "end"]).sort_values(["start", "end"])

# Auto-class controls
st.sidebar.markdown("---")
st.sidebar.write("**Auto-classification** (used when no valid classes CSV)")
auto_win = st.sidebar.number_input("Window (minutes)", min_value=1, max_value=30, value=5, step=1)
auto_overlap = st.sidebar.slider("Window overlap", 0.0, 0.9, 0.5, 0.05)

# Display options
st.sidebar.markdown("---")
smooth_min = st.sidebar.checkbox("Show 5-min rolling mean", value=True)
show_points = st.sidebar.checkbox("Show raw points", value=False)
opacity_bands = st.sidebar.slider("Class shading opacity", 0.05, 0.6, 0.30, 0.01)

st.sidebar.markdown("---")
st.sidebar.write("**Class visibility**")
visible_classes = {k: st.sidebar.checkbox(k, value=(k != "UNCERTAIN")) for k in CLASS_COLORS.keys()}

# Time window
st.markdown("### Time Range")
tmin, tmax = ts.index.min(), ts.index.max()
start, end = st.slider(
    "Select time window",
    min_value=tmin.to_pydatetime(),
    max_value=tmax.to_pydatetime(),
    value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
    step=timedelta(minutes=1),
    format="MM/DD HH:mm",
)

view = ts.loc[(ts.index >= start) & (ts.index <= end), ["Ovality"]].copy()
if smooth_min:
    view["smooth_5min"] = view["Ovality"].rolling("5min", min_periods=1).mean()

# Decide which classes to use
if not classes_ext.empty:
    classes = classes_ext
else:
    classes = compute_classes_time_windows(ts["Ovality"], win_min=int(auto_win), overlap=float(auto_overlap))

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics", expanded=False):
    st.write(f"Total points loaded: {len(ts)}")
    if classes is not None and len(classes):
        st.write("Class label counts:", classes["label"].value_counts().to_dict())
        st.dataframe(classes.head(5))
    else:
        st.warning("No classes available (nothing to highlight). Try lowering the window size or overlap.")

# ---------------- Main plot ----------------
fig = go.Figure()
if show_points:
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["Ovality"],
            mode="markers",
            name="raw",
            marker=dict(size=3),
            opacity=0.6,
        )
    )
else:
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["Ovality"],
            mode="lines",
            name="raw",
            line=dict(width=1),
            opacity=0.6,
        )
    )

if "smooth_5min" in view:
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["smooth_5min"],
            mode="lines",
            name="smooth (5 min)",
            line=dict(width=3),
        )
    )

# Overlays
if classes is not None and len(classes):
    csel = classes[(classes["end"] >= pd.Timestamp(start)) & (classes["start"] <= pd.Timestamp(end))].copy()
    for _, row in csel.iterrows():
        label = normalize_label(row.get("label", "UNCERTAIN"))
        if not visible_classes.get(label, True):
            continue
        x0 = max(pd.Timestamp(start), row["start"])
        x1 = min(pd.Timestamp(end), row["end"])
        if x1 <= x0:
            continue
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=CLASS_COLORS.get(label, "#BBBBBB"),
            opacity=opacity_bands,
            line_width=0,
            layer="below",
            annotation_text=label,
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#333"),
        )

fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    title="Ovality — with class segments",
    xaxis_title="time",
    yaxis_title="Ovality",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Class barcode ----------------
st.markdown("#### Class Timeline")
if classes is not None and len(classes):
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="1min")
    if len(idx):
        order = list(CLASS_COLORS.keys())
        code_map = {c: i for i, c in enumerate(order)}
        grid = pd.DataFrame(index=idx)
        grid["code"] = np.nan

        for _, r in classes.iterrows():
            lbl = normalize_label(r.get("label", "UNCERTAIN"))
            if lbl not in code_map or not visible_classes.get(lbl, True):
                continue
            s_ = max(pd.Timestamp(start), r["start"])
            e_ = min(pd.Timestamp(end), r["end"])
            if e_ <= s_:
                continue
            grid.loc[s_:e_, "code"] = code_map[lbl]

        colors = ["rgba(0,0,0,0)"] * len(grid)
        vals = grid["code"].values
        for cls_name, idx_code in code_map.items():
            if not visible_classes.get(cls_name, True):
                continue
            color = CLASS_COLORS[cls_name]
            for i, v in enumerate(vals):
                if not np.isnan(v) and int(v) == idx_code:
                    colors[i] = color

        tl = go.Figure(
            go.Bar(
                x=grid.index,
                y=np.ones(len(grid.index)),
                marker_color=colors,
                marker_line_width=0,
                hovertext=[str(t) for t in grid.index],
                hoverinfo="text",
            )
        )
        tl.update_yaxes(visible=False)
        tl.update_xaxes(title="time")
        tl.update_layout(height=120, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(tl, use_container_width=True)
else:
    st.info("No class rows to render.")
