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

# ---------- NEW: Detect/derive metric type (Patient-centred vs Team-centred) ----------
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
    # If clearly team and not obviously patient
    if any(k in t for k in _team_kw) and not any(k in t for k in _patient_kw):
        return "Team-centred"
    # If clearly patient and not obviously team
    if any(k in t for k in _patient_kw) and not any(k in t for k in _team_kw):
        return "Patient-centred"
    # Heuristic fallback: favour patient-centred as default for ambiguous KPI text
    return "Patient-centred"

def derive_metric_type(frame: pd.DataFrame) -> pd.Series:
    if TYPE_COL:
        # Use the discovered column if it contains recognisable values
        vals = frame[TYPE_COL].apply(_normalise_type_value)
        if vals.notna().any():
            # If some are NA, fill with heuristic
            base = vals.copy()
            mask_na = base.isna()
            if mask_na.any():
                text = (frame["Item Description"].fillna("").astype(str) + " | " +
                        frame["Metric"].fillna("").astype(str))
                base[mask_na] = text[mask_na].apply(_classify_by_keywords)
            return base
    # No usable column found → keyword heuristic on description + metric code
    text = (frame["Item Description"].fillna("").astype(str) + " | " +
            frame["Metric"].fillna("").astype(str))
    return text.apply(_classify_by_keywords)

df["MetricType"] = derive_metric_type(df)

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")

    # NEW: Metric set selector
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

# Apply metric-set filter
if metric_set != "All":
    f = f[f["MetricType"] == metric_set]

f["Item Description"] = f["Item Description"].fillna("")
f["MetricLabel"] = f.apply(
    lambda r: f"{r['Metric']} — {r['Item Description']}" if r["Item Description"] else str(r["Metric"]),
    axis=1,
)

if q:
    qlow = q.lower()
    # IMPORTANT: regex=False so literal search works with dots like "1.1"
    f = f[f["MetricLabel"].str.lower().str.contains(qlow, regex=False)]

# ---------- Metric selection (sorted numerically) ----------
# Build label -> metric map
metric_map = dict(zip(f_





