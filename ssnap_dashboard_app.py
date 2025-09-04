import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import altair as alt

# Optional: PNG export
try:
    from altair_saver import save as altair_save
    ALT_SAVE_AVAILABLE = True
except Exception:
    ALT_SAVE_AVAILABLE = False

st.set_page_config(page_title="SSNAP – Key Indicators (multi-site workbook)", layout="wide")
st.title("SSNAP Key indicators, PRUH & DH HASU/SU")

# ---------- Data loading (cached, no upload) ----------
DEFAULT_XLSX = Path("Key indicators DH-PRUH.xlsx")

@st.cache_data(show_spinner=False)
def read_all_sheets_cached(pathlike: Path) -> pd.DataFrame:
    """Read all sheets from the Excel file and add a 'Site' column = sheet name."""
    if not pathlike.exists():
        raise FileNotFoundError(f"File not found: {pathlike.resolve()}")
    xls = pd.ExcelFile(pathlike)
    frames = []
    for sh in xls.sheet_names:
        df_sh = pd.read_excel(pathlike, sheet_name=sh)
        df_sh["Site"] = sh  # derive Site from sheet name
        frames.append(df_sh)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    df = read_all_sheets_cached(DEFAULT_XLSX)
except Exception as e:
    st.error(f"Failed to load workbook: {e}")
    st.stop()

# ---------- Validate ----------
required_cols = {"Metric", "Item Description", "Site"}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    st.error(f"Expected columns missing: {', '.join(sorted(missing))}")
    st.stop()

# Identify year columns (e.g., 2013/14 ... 2023/24)
year_cols = [c for c in df.columns if re.fullmatch(r"(19|20)\d{2}/\d{2}", str(c).strip())]
year_cols = sorted(year_cols, key=lambda x: int(x.split("/")[0]))
if not year_cols:
    st.error("No financial year columns found (e.g., 2019/20).")
    st.stop()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    sites = sorted(df["Site"].dropna().unique())
    selected_sites = st.multiselect("Site(s)", options=sites, default=sites)
    selected_years = st.multiselect("Years", options=year_cols, default=year_cols)
    q = st.text_input("Search metrics/descriptions", placeholder="e.g., thrombolysis, 1.1A, swallow...")

# Apply filters
f = df[df["Site"].isin(selected_sites)].copy() if selected_sites else df.copy()
f["Item Description"] = f["Item Description"].fillna("")
f["MetricLabel"] = f.apply(
    lambda r: f"{r['Metric']} — {r['Item Description']}" if r["Item Description"] else str(r["Metric"]),
    axis=1,
)

if q:
    qlow = q.lower()
    f = f[f["MetricLabel"].str.lower().str_contains(qlow, regex=False)]

# ---------- Metric selection ----------
metric_map = dict(zip(f["MetricLabel"], f["Metric"]))
labels_sorted = sorted(metric_map.keys())
if not labels_sorted:
    st.warning("No rows match your filters/search.")
    st.stop()

selected_label = st.selectbox("Select a metric", options=labels_sorted)
selected_metric = metric_map[selected_label]
mf = f[f["Metric"] == selected_metric].copy()

display_years = selected_years if selected_years else year_cols
table_cols = ["Metric", "Item Description", "Site"] + display_years

st.subheader("Filtered table")
st.dataframe(mf[table_cols].reset_index(drop=True), use_container_width=True, height=350)

# ---------- Long format for chart ----------
def parse_value_to_minutes_or_number(v):
    """Return minutes since midnight if HH:MM(/:SS), else numeric."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        try:
            parts = list(map(int, s.split(":")))
            if len(parts) == 2:
                h, m = parts; sec = 0
            else:
                h, m, sec = parts
            return h * 60 + m + sec / 60.0
        except Exception:
            return np.nan
    return pd.to_numeric(s, errors="coerce")

long = mf.melt(
    id_vars=["Metric", "Item Description", "Site"],
    value_vars=display_years,
    var_name="Year",
    value_name="Raw"
)
long["Value"] = long["Raw"].apply(parse_value_to_minutes_or_number)
long["Year"] = pd.Categorical(long["Year"], categories=year_cols, ordered=True)

# Detect time-of-day style (HH:MM/HH:MM:SS)
is_timeofday = long["Raw"].dropna().astype(str).str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").all()

# ---------- Colour mapping (DH → blues, PRUH → pinks, others → greys) ----------
def site_color_scale(sites_in_plot):
    blues = ["#1f77b4", "#2a9df4", "#1b6ca8", "#74add1", "#a6cee3"]
    pinks = ["#e377c2", "#ff6fb5", "#d45087", "#fb9a99", "#f781bf"]
    greys = ["#7f7f7f", "#aaaaaa", "#555555"]

    domain = []
    colors = []
    bi = pi = gi = 0
    for s in sites_in_plot:
        sl = str(s).lower()
        if "pruh" in sl:
            domain.append(s); colors.append(pinks[pi % len(pinks)]); pi += 1
        elif sl.startswith("dh") or " dh" in sl or "dh " in sl:
            domain.append(s); colors.append(blues[bi % len(blues)]); bi += 1
        else:
            domain.append(s); colors.append(greys[gi % len(greys)]); gi += 1
    return domain, colors

sites_in_plot = list(pd.unique(long["Site"]))
sites_in_plot.sort()
color_domain, color_range = site_color_scale(sites_in_plot)
color_enc = alt.Color(
    "Site:N",
    legend=alt.Legend(title="Site"),
    scale=alt.Scale(domain=color_domain, range=color_range)
)

# ---------- Helpers ----------
def minute_ticks(ymin, ymax, max_ticks=8):
    if ymin is None or ymax is None:
        return [0, 360, 720, 1080, 1440]
    span = max(1, ymax - ymin)
    for step in [30, 60, 120, 180, 240, 360]:
        if span / step <= max_ticks:
            break
    start = int(np.floor(ymin / step) * step)
    end = int(np.ceil(ymax / step) * step)
    vals = list(range(start, end + step, step))
    vals = [v for v in vals if 0 <= v <= 1440]
    return vals or [int(ymin), int(ymax)]

# ---------- Chart ----------
st.subheader("Trend chart")

# Zoom controls
y_min, y_max = None, None
if st.checkbox("Zoom Y-axis"):
    col1, col2 = st.columns(2)
    valid_vals = long["Value"].dropna()
    default_min = float(valid_vals.min()) if not valid_vals.empty else 0.0
    default_max = float(valid_vals.max()) if not valid_vals.empty else 1.0
    with col1:
        y_min = st.number_input("Y-axis min", value=default_min, step=1.0)
    with col2:
        y_max = st.number_input("Y-axis max", value=default_max, step=1.0)

if is_timeofday:
    # Default domain: 00:00 to (max + 1 hour), capped at 24:00, unless user zooms
    valid_vals = long["Value"].dropna()
    if y_min is None or y_max is None:
        max_val = float(valid_vals.max()) if not valid_vals.empty else 0.0
        default_upper = min(1440.0, (np.ceil(max_val / 60.0) * 60.0) + 60.0)
        domain = [0.0, default_upper]
    else:
        domain = [y_min, y_max]

    tick_vals = minute_ticks(domain[0], domain[1])
    y_scale = alt.Scale(domain=domain)
    y_axis = alt.Axis(
        orient="left",
        values=tick_vals,
        labelExpr="timeFormat(datum.value*60*1000, '%H:%M')",
        title="Time (HH:MM)",
        grid=True
    )
    line = (
        alt.Chart(long.dropna(subset=["Value"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:N", sort=year_cols, title="Year"),
            y=alt.Y("Value:Q", scale=y_scale, axis=y_axis),
            color=color_enc,
            tooltip=["Metric", "Item Description", "Site", "Year", "Raw"]
        )
        .properties(height=450)
    )
else:
    # Default non-time axis: Percentage
    y_scale = alt.Scale(domain=[y_min, y_max]) if (y_min is not None and y_max is not None) else alt.Undefined
    y_axis = alt.Axis(orient="left", title="Percentage", grid=True)
    line = (
        alt.Chart(long.dropna(subset=["Value"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:N", sort=year_cols, title="Year"),
            y=alt.Y("Value:Q", scale=y_scale, axis=y_axis),
            color=color_enc,
            tooltip=["Metric", "Item Description", "Site", "Year", "Raw", "Value"]
        )
        .properties(height=450)
    )

st.altair_chart(line, use_container_width=True)

# ---------- Downloads ----------
st.download_button(
    "Download filtered table (CSV)",
    mf[table_cols].to_csv(index=False).encode("utf-8"),
    file_name="key_indicators_filtered_table.csv",
    mime="text/csv"
)

st.download_button(
    "Download chart data (CSV)",
    long.to_csv(index=False).encode("utf-8"),
    file_name="key_indicators_chart_long.csv",
    mime="text/csv"
)

# ---------- Chart PNG export with fallback ----------
if ALT_SAVE_AVAILABLE:
    try:
        buf = io.BytesIO()
        altair_save(line, fp=buf, fmt="png")  # auto backend
        st.download_button("Download chart (PNG)", data=buf.getvalue(), file_name="chart.png", mime="image/png")
    except Exception:
        try:
            buf = io.BytesIO()
            altair_save(line, fp=buf, fmt="png", method="node")  # node fallback
            st.download_button("Download chart (PNG)", data=buf.getvalue(), file_name="chart.png", mime="image/png")
        except Exception as e2:
            st.warning(
                "PNG export failed. Please ensure the following are installed/updated:\n"
                "  pip install -U altair_saver vl-convert-python\n\n"
                f"Details: {e2}"
            )
else:
    st.info("To enable PNG downloads, install:\n    pip install altair_saver vl-convert-python")


