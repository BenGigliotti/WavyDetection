# waviness_dashboard.py
# Streamlit app to visualize a signal with waviness class overlays
# Run: streamlit run waviness_dashboard.py

import os, glob
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Utilities ----------------------------- #

CLASS_COLORS = {
    "STEADY":        "#4CAF50",
    "MILD_WAVE":     "#FFB300",
    "STRONG_WAVE":   "#E53935",
    "DRIFT":         "#2196F3",
    "BURSTY_NOISY":  "#8E24AA",
    "UNCERTAIN":     "#9E9E9E",
}

def resolve_path(p: str):
    """Return an existing path for p by trying a few likely locations."""
    if not p:
        return None
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    candidates = [
        os.path.join("ndc_0926_outputs", base),
        os.path.join(os.getcwd(), base),
        os.path.join(os.getcwd(), "ndc_0926_outputs", base),
        os.path.join(os.path.dirname(__file__), base) if "__file__" in globals() else None,
        os.path.join(os.path.dirname(__file__), "ndc_0926_outputs", base) if "__file__" in globals() else None,
        os.path.join(os.path.dirname(os.getcwd()), "ndc_0926_outputs", base),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def suggest_candidates(pattern="**/merged_*.csv"):
    hits = glob.glob(pattern, recursive=True)
    return sorted(hits)

def normalize_label(s: str) -> str:
    """Map various free-text labels to canonical keys used for coloring."""
    if not isinstance(s, str):
        return "UNCERTAIN"
    t = s.strip().upper().replace("-", "_").replace(" ", "_").replace("/", "_")
    alias = {
        "BURSTY_NOISY": "BURSTY_NOISY",
        "BURSTY_NOISE": "BURSTY_NOISY",
        "NOISY": "BURSTY_NOISY",
        "MILD": "MILD_WAVE",
        "MILDWAVE": "MILD_WAVE",
        "STRONG": "STRONG_WAVE",
        "STRONGWAVE": "STRONG_WAVE",
        "STEADY": "STEADY",
        "DRIFT": "DRIFT",
        "UNCERTAIN": "UNCERTAIN",
    }
    return alias.get(t, t)

# ----------------------------- Loaders ----------------------------- #

@st.cache_data(show_spinner=False)
def load_timeseries(path: str, ts_col: str) -> pd.DataFrame:
    """Load merged time series CSV and coerce numeric columns."""
    rp = resolve_path(path)
    if not rp:
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(rp, parse_dates=[ts_col]).sort_values(ts_col).set_index(ts_col)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_classes(path: str, series_start: pd.Timestamp, series_end: pd.Timestamp) -> pd.DataFrame:
    """Load class windows CSV and normalize columns/labels, clip to series range."""
    rp = resolve_path(path)
    if not rp:
        raise FileNotFoundError(f"Not found: {path}")
    c = pd.read_csv(rp)

    # Map flexible column names -> start, end, label
    colmap = {k.lower(): k for k in c.columns}
    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    start_col = pick("start", "window_start", "timestamp", "time_start")
    end_col   = pick("end", "window_end", "time_end", "stop")
    label_col = pick("label", "class", "wave_class")

    if not start_col:
        raise ValueError("Classes CSV missing a start column (e.g., start/window_start/timestamp).")
    if not label_col:
        # will create UNCERTAIN if missing, but warn
        label_col = None

    c["start"] = pd.to_datetime(c[start_col], errors="coerce")
    if end_col:
        c["end"] = pd.to_datetime(c[end_col], errors="coerce")
    else:
        # Infer end as next start; last uses median gap or 5 min default
        c = c.sort_values("start")
        gaps = c["start"].diff().dropna()
        win = gaps.median() if len(gaps) else pd.Timedelta(minutes=5)
        c["end"] = c["start"].shift(-1)
        c.loc[c["end"].isna(), "end"] = c["start"] + win

    # Label handling
    if label_col:
        c["label"] = c[label_col].astype(str).map(normalize_label)
    else:
        c["label"] = "UNCERTAIN"

    # Drop bad rows and clip to series span
    c = c.dropna(subset=["start", "end"])
    c = c[c["end"] > c["start"]]
    if pd.notna(series_start) and pd.notna(series_end):
        c["start"] = c["start"].clip(lower=series_start, upper=series_end)
        c["end"]   = c["end"].clip(lower=series_start, upper=series_end)
        c = c[c["end"] > c["start"]]
    c = c.sort_values(["start", "end"]).reset_index(drop=True)
    return c

# ----------------------------- UI ----------------------------- #

st.set_page_config(page_title="Waviness Dashboard", layout="wide")

st.sidebar.header("Inputs")
merged_path = st.sidebar.text_input(
    "Merged time-series CSV (with timestamp + tags)",
    value="ndc_0926_outputs/merged_2025-09-26.csv",
)
classes_path = st.sidebar.text_input(
    "Waves classes CSV (start, end, label)",
    value="ndc_0926_outputs/waves_classes_NDC_System_Ovality_Value__Tag_value.csv",
)
features_path = st.sidebar.text_input(
    "Waves features CSV (optional)",
    value="ndc_0926_outputs/waves_features_NDC_System_Ovality_Value__Tag_value.csv",
)
ts_col = st.sidebar.text_input("Timestamp column name", value="timestamp")
default_tag = st.sidebar.text_input("Signal to visualize (column name)", value="NDC_System_Ovality_Value__Tag_value")

st.sidebar.markdown("---")
st.sidebar.write("**Display options**")
smooth_min = st.sidebar.checkbox("Show 5-min rolling mean", value=True)
show_points = st.sidebar.checkbox("Show raw points", value=False)
opacity_bands = st.sidebar.slider("Class shading opacity", 0.05, 0.60, 0.30, 0.01)

st.sidebar.markdown("---")
st.sidebar.write("**Class visibility**")
visible_classes = {k: st.sidebar.checkbox(k, value=(k != "UNCERTAIN")) for k in CLASS_COLORS.keys()}

# ----------------------------- Load data ----------------------------- #

ts = None
classes = None
features = None
load_err = False

try:
    ts = load_timeseries(merged_path, ts_col)
except Exception as e:
    st.error(f"Failed to load timeseries: {e}")
    st.info(f"Working directory: {os.getcwd()}")
    st.info("Try one of these found files:")
    st.code("\n".join(suggest_candidates()), language="text")
    load_err = True

if not load_err:
    # Load classes AFTER we know series span
    try:
        classes = load_classes(classes_path, series_start=ts.index.min(), series_end=ts.index.max())
    except Exception as e:
        st.warning(f"Could not load classes file: {e}")
        classes = None

    # Features are optional
    try:
        rp = resolve_path(features_path)
        if rp:
            features = pd.read_csv(rp, parse_dates=["start", "end"])
    except Exception:
        features = None

if load_err:
    st.stop()

# ----------------------------- Tag & time filters ----------------------------- #

numeric_cols = [c for c in ts.columns if pd.api.types.is_numeric_dtype(ts[c])]
if not numeric_cols:
    st.error("No numeric columns detected in the merged CSV.")
    st.stop()

if default_tag not in numeric_cols:
    default_tag = numeric_cols[0]

tag = st.selectbox("Signal to plot:", options=numeric_cols, index=numeric_cols.index(default_tag))

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

mask = (ts.index >= start) & (ts.index <= end)
view = ts.loc[mask, [tag]].copy()
if len(view) == 0:
    st.info("No data in selected window.")
    st.stop()

if smooth_min:
    # use '5min' to avoid future warning for '5T'
    view["smooth_5min"] = view[tag].rolling("5min", min_periods=1).mean()

# ----------------------------- Diagnostics ----------------------------- #

with st.expander("Class diagnostics", expanded=False):
    if classes is None or classes.empty:
        st.write("No class rows loaded.")
    else:
        st.write("Unique labels:", sorted(classes["label"].unique()))
        overlap = classes[(classes["end"] >= pd.Timestamp(start)) & (classes["start"] <= pd.Timestamp(end))]
        st.write(f"Overlapping segments in current window: {len(overlap)}")
        if len(overlap):
            st.dataframe(overlap.head(10))

# ----------------------------- Main plot ----------------------------- #

fig = go.Figure()

if show_points:
    fig.add_trace(go.Scatter(
        x=view.index, y=view[tag],
        mode="markers", name="raw",
        marker=dict(size=3),
        opacity=0.6
    ))
else:
    fig.add_trace(go.Scatter(
        x=view.index, y=view[tag],
        mode="lines", name="raw",
        line=dict(width=1),
        opacity=0.6
    ))

if smooth_min and "smooth_5min" in view:
    fig.add_trace(go.Scatter(
        x=view.index, y=view["smooth_5min"],
        mode="lines", name="smooth (5 min)",
        line=dict(width=3)
    ))

# Colored rectangles for classes
if classes is not None and len(classes):
    csel = classes[(classes["end"] >= pd.Timestamp(start)) & (classes["start"] <= pd.Timestamp(end))].copy()
    for _, row in csel.iterrows():
        label = str(row.get("label", "UNCERTAIN"))
        if not visible_classes.get(label, True):
            continue
        x0 = max(pd.Timestamp(start), row["start"])
        x1 = min(pd.Timestamp(end), row["end"])
        if x1 <= x0:
            continue
        color = CLASS_COLORS.get(label, "#BBBBBB")
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color,
            opacity=opacity_bands,
            line_width=0,
            layer="below",
            annotation_text=label,
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#333")
        )

fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    title=f"{tag} — w/ class segments",
    xaxis_title="time",
    yaxis_title=tag,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Mini timeline ----------------------------- #

st.markdown("#### Class Timeline")
if classes is not None and len(classes):
    # minute grid to show a barcode of classes
    grid_index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="1min")
    if len(grid_index) > 0:
        grid = pd.DataFrame(index=grid_index)
        grid["code"] = np.nan
        order = list(CLASS_COLORS.keys())
        code_map = {c: i for i, c in enumerate(order)}

        for _, row in classes.iterrows():
            lbl = str(row.get("label", "UNCERTAIN"))
            if lbl not in code_map or not visible_classes.get(lbl, True):
                continue
            s = max(pd.Timestamp(start), row["start"])
            e = min(pd.Timestamp(end), row["end"])
            if e <= s:
                continue
            grid.loc[s:e, "code"] = code_map[lbl]

        # make color band
        colors = ["rgba(0,0,0,0)"] * len(grid)
        for cls_name, idx in code_map.items():
            if not visible_classes.get(cls_name, True):
                continue
            c = CLASS_COLORS[cls_name]
            mask_cls = (grid["code"].values == idx)
            for i, ok in enumerate(mask_cls):
                if ok:
                    colors[i] = c

        tl = go.Figure(go.Bar(
            x=grid.index, y=np.ones(len(grid.index)),
            marker_color=colors, marker_line_width=0,
            hovertext=[str(t) for t in grid.index],
            hoverinfo="text",
        ))
        tl.update_yaxes(visible=False)
        tl.update_xaxes(title="time")
        tl.update_layout(height=120, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(tl, use_container_width=True)
    else:
        st.info("Selected time window is empty.")
else:
    st.info("No class file loaded or it contains no rows.")

# ----------------------------- Features table ----------------------------- #

st.markdown("#### Window Features & Labels")
if features is not None and len(features):
    fsel = features[(features["end"] >= pd.Timestamp(start)) & (features["start"] <= pd.Timestamp(end))].copy()
    if "label" in fsel.columns:
        fsel["label"] = fsel["label"].astype(str).map(normalize_label)
        chosen = st.multiselect("Filter by class",
                                options=list(CLASS_COLORS.keys()),
                                default=[k for k, v in visible_classes.items() if v])
        if chosen:
            fsel = fsel[fsel["label"].isin(chosen)]
    keep_cols = ["start", "end", "label", "ptp_amp", "std", "rms",
                 "max_acf", "dom_power_ratio", "slope", "spectral_entropy", "acf_lag", "acf_lag_min"]
    show_cols = [c for c in keep_cols if c in fsel.columns]
    st.dataframe(fsel[show_cols].sort_values("start"), use_container_width=True)
else:
    st.info("No features file loaded or empty.")
