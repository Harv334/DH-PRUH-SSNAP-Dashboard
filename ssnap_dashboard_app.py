import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import altair as alt

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

# ---------- Metric type (Patient-centred vs Team-centred) ----------
def _find_type_column(cols):
    """Try to find a column that encodes metric type."""
    patterns = [
        r"(metric|indicator)\s*(type|category)",
        r"(patient|team).*(type|category|cent(er|re)d)",
        r"(patient\s*/\s*team)",
        r"(patient|team)$",
        r"(cent(er|re)d)"
    ]
    for c in cols:
        cl = c.lower()
        if any(re.search(p, cl) for p in patterns):
            return c
    return None

TYPE_COL = _find_type_column(df.columns)

_patient_kw = [
    "door-to-needle", "thrombolysis", "thrombectomy", "swallow", "screen",
    "admitted", "brain imaging", "ct", "mri", "arrive", "discharge", "mortality",
    "palliative", "length of stay", "los", "independence", "therapy goals",
    "anticoag", "antiplate", "statin", "bp", "blood pressure", "home", "rehab",
    "nutrition", "depression", "mood", "continence", "carer", "follow-up"
]
_team_kw = [
    "staff", "staffing", "consultant", "senior decision", "ward round",
    "nurse", "nursing", "establishment", "rota", "seven-day", "7 day", "7-day",
    "therapy minutes", "therapy hours", "physio", "ot ", "occupational therapy",
    "slt", "speech and language", "multidisciplinary", "mdt", "weekend cover",
    "specialist intensity", "team"
]

def _normalise_type_value(v):
    s = str(v).strip().lower()
    if "patient" in s:
        return "Patient-centred"
    if "team" in s:
        return "Team-centred"
    return None

def _classify_by_keywords(text):
    t = str(text or "").lower()
    if any(k in t for k in _team_kw) and not any(k in t for k in _patient_kw):
        return "Team-centred"
    if any(k in t for k in _patient_kw) and not any(k in t for k in _team_kw):
        return "Patient-centred"
    return "Patient-centred"

def derive_metric_type(frame: pd.DataFrame) -> pd.Series:
    if TYPE_COL:
        vals = frame[TYPE_COL].apply(_normalise_type_value)
        if vals.notna().any():
            base = vals.copy()
            mask_na = base.isna()
            if mask_na.any():
                text = (frame["Item Description"].fillna("").astype(str) + " | " +
                        frame["Metric"].fillna("").astype(str))
                base[mask_na] = text[mask_na].apply(_classify_by_keywords)
            return base
    text = (frame["Item Description"].fillna("").astype(str) + " | " +
            frame["Metric"].fillna("").astype(str))
    return text.apply(_classify_by_keywords)

df["MetricType"] = derive_metric_type(df)

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")

    metric_set = st.radio(
        "Metric set",
        options=["All", "Patient-centred", "Team-centred"],
        horizontal=True,
        index=0
    )

    sites = sorted(df["Site"].dropna().unique())
    selected_sites = st.multiselect("Site(s)", options=sites, default=sites)
    selected_years = st.multiselect("Years", options=year_cols, default=year_cols)
    q = st.text_input("Search metrics/descriptions", placeholder="e.g., thrombolysis, 1.1A, swallow...")

# Apply filters
f = df.copy()
if selected_sites:
    f = f[f["Site"].isin(selected_sites)]

if metric_set != "All":
    f = f[f["MetricType"] == metric_set]

f["Item Description"] = f["Item Description"].fillna("")
f["MetricLabel"] = f.apply(
    lambda r: f"{r['Metric']} — {r['Item Description']}" if r["Item Description"] else str(r["Metric"]),
    axis=1,
)

if q:
    qlow = q.lower()
    f = f[f["MetricLabel"].str.lower().str.contains(qlow, regex=False)]

# ---------- Metric selection (sorted numerically) ----------
metric_map = dict(zip(f["MetricLabel"], f["Metric"]))

def metric_sort_key(metric_str: str):
    s = (metric_str or "").strip()
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*([A-Za-z]?)", s)
    if not m:
        return (10**9, 10**9, 10**9, 10**9, s.lower())
    major = int(m.group(1)) if m.group(1) else 0
    minor = int(m.group(2)) if m.group(2) else 0
    sub   = int(m.group(3)) if m.group(3) else 0
    letter = m.group(4).upper() if m.group(4) else ""
    letter_rank = (ord(letter) - 64) if letter else 0
    return (major, minor, sub, letter_rank, s.lower())

labels_sorted = sorted(metric_map.keys(), key=lambda lab: metric_sort_key(metric_map[lab]))
if not labels_sorted:
    st.warning("No rows match your filters/search.")
    st.stop()

selected_label = st.selectbox("Select a metric", options=labels_sorted)
selected_metric = metric_map[selected_label]
mf = f[f["Metric"] == selected_metric].copy()

display_years = selected_years if selected_years else year_cols
table_cols = ["MetricType", "Metric", "Item Description", "Site"] + display_years

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
    id_vars=["MetricType", "Metric", "Item Description", "Site"],
    value_vars=display_years,
    var_name="Year",
    value_name="Raw"
)
long["Value"] = long["Raw"].apply(parse_value_to_minutes_or_number)
long["Year"] = pd.Categorical(long["Year"], categories=year_cols, ordered=True)

# Detect time-of-day style
is_timeofday = long["Raw"].dropna().astype(str).str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").all()

# ---------- Colour mapping ----------
def site_color_scale(sites_in_plot):
    blues = ["#1f77b4", "#2a9df4", "#1b6ca8", "#74add1", "#a6cee3"]
    pinks = ["#e377c2", "#ff6fb5", "#d45087", "#fb9a99", "#f781bf"]
    greys = ["#7f7f7f", "#aaaaaa", "#555555"]

    domain, colors = [], []
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

desc_candidates = (
    mf["Item Description"].dropna().astype(str).str.strip().unique().tolist()
    if "Item Description" in mf.columns else []
)
item_desc = next((d for d in desc_candidates if d), "")
if item_desc:
    st.markdown(f"**Item description:** {item_desc}")
    st.caption(f"Metric set: **{mf['MetricType'].iloc[0]}**")

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
            tooltip=["MetricType", "Metric", "Item Description", "Site", "Year", "Raw"]
        )
        .properties(height=450)
    )
else:
    y_scale = alt.Scale(domain=[y_min, y_max]) if (y_min is not None and y_max is not None) else alt.Undefined
    y_axis = alt.Axis(orient="left", title="Percentage", grid=True)
    line = (
        alt.Chart(long.dropna(subset=["Value"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:N", sort=year_cols, title="Year"),
            y=alt.Y("Value:Q", scale=y_scale, axis=y_axis),
            color=color_enc,
            tooltip=["MetricType", "Metric", "Item Description", "Site", "Year", "Raw", "Value"]
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







