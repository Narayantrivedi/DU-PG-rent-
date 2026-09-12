"""
Keshav Mahavidyalaya PG Finder & Rent Estimator
--------------------------------------------------
A calm, Android-style Streamlit app to browse PG (paying-guest) listings
around Pitampura / Keshav Mahavidyalaya and estimate a fair monthly rent.

Run with:  streamlit run pg_finder_app.py
"""

import math
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KM PG Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    bg, card_bg, text_color, sub_text, border_color = "#14161a", "#1d2024", "#e8e6e1", "#9a9690", "#2a2d32"
    banner_bg_1, banner_bg_2 = "#1c2b27", "#20241c"
else:
    bg, card_bg, text_color, sub_text, border_color = "#faf8f5", "#ffffff", "#26241f", "#726e66", "#e6e2da"
    banner_bg_1, banner_bg_2 = "#eef3ee", "#f4efe6"

PRIMARY = "#5b7f70"       # muted sage — calmer than a bright Material teal
PRIMARY_DARK = "#3f5a4e"
ACCENT_WARM = "#c98a58"   # warm terracotta accent, used sparingly

# Soft, natural palette reused across all charts so they feel part of the same app
CHART_PALETTE = ["#5b7f70", "#c98a58", "#8ea9c9", "#c97a72", "#a98fc9", "#9db17c"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }}

    .stApp {{
        background: {bg};
        color: {text_color};
    }}

    /* ---- header banner ---- */
    .campus-banner {{
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid {border_color};
        margin-top: 0.6rem;
        margin-bottom: 1.3rem;
        background: linear-gradient(120deg, {banner_bg_1}, {banner_bg_2});
    }}
    .campus-banner img {{
        width: 100%;
        max-height: 260px;
        object-fit: cover;
        display: block;
    }}
    .campus-banner-overlay {{
        position: absolute;
        left: 0; right: 0; bottom: 0;
        padding: 0.9rem 1.3rem;
        background: linear-gradient(to top, rgba(20,20,18,0.55), rgba(20,20,18,0));
    }}
    .campus-banner-overlay h1 {{
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
        color: #ffffff;
        letter-spacing: 0.2px;
    }}
    .campus-banner-overlay p {{
        margin: 0.15rem 0 0 0;
        color: #f1efe9;
        font-size: 0.88rem;
        opacity: 0.92;
    }}

    .page-title {{
        padding: 0 0 0.5rem 0;
        border-bottom: 1px solid {border_color};
        margin-bottom: 1.2rem;
    }}
    .page-title h1 {{
        font-size: 1.35rem;
        font-weight: 600;
        margin: 0;
        color: {text_color};
    }}
    .page-title p {{
        margin: 0.25rem 0 0.6rem 0;
        color: {sub_text};
        font-size: 0.92rem;
    }}

    .metric-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }}
    .metric-card h3 {{ margin: 0; font-weight: 600; color: {PRIMARY_DARK}; }}
    .metric-card p {{ margin: 0.15rem 0 0 0; color: {sub_text}; font-size: 0.78rem; letter-spacing: 0.2px; }}

    .pg-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        color: {text_color};
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    .chip {{
        display: inline-block;
        padding: 3px 11px;
        border-radius: 20px;
        font-size: 0.74rem;
        margin-right: 6px;
        margin-bottom: 4px;
        background: {border_color};
        color: {text_color};
        font-weight: 500;
    }}
    .chip-verified {{ background: #e3ede6; color: #3f5a4e; }}
    .chip-lowrooms {{ background: #f6e9dd; color: #9a5a2a; }}
    .rent-tag {{ font-size: 1.12rem; font-weight: 600; color: {PRIMARY_DARK}; }}
    .review-box {{
        background: {bg};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.4rem;
        font-size: 0.86rem;
    }}
    .section-note {{
        color: {sub_text};
        font-size: 0.88rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }}

    .stButton > button {{
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER — banner with campus image + title
# ============================================================
def _load_campus_image_data_uri():
    """
    Looks for a real campus photo dropped next to this script
    (e.g. keshav_mahavidyalaya.jpg / .png). If present, it's used as
    the banner. Otherwise we fall back to a calm, hand-drawn campus
    illustration further below, so the app looks finished either way.
    """
    candidates = ["keshav_mahavidyalaya.jpg", "keshav_mahavidyalaya.jpeg",
                  "keshav_mahavidyalaya.png", "campus_photo.jpg", "campus_photo.png"]
    here = Path(__file__).resolve().parent
    for name in candidates:
        p = here / name
        if p.exists():
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{encoded}"
    return None


CAMPUS_ILLUSTRATION_SVG = f"""
<svg viewBox="0 0 900 260" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice"
     style="width:100%; height:100%; display:block;">
  <rect width="900" height="260" fill="url(#skyGrad)"/>
  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{banner_bg_1}"/>
      <stop offset="100%" stop-color="{banner_bg_2}"/>
    </linearGradient>
  </defs>
  <ellipse cx="150" cy="230" rx="420" ry="60" fill="{PRIMARY}" opacity="0.10"/>
  <ellipse cx="720" cy="245" rx="360" ry="50" fill="{ACCENT_WARM}" opacity="0.10"/>

  <!-- college building -->
  <g transform="translate(300,60)">
    <rect x="0" y="70" width="300" height="110" fill="{PRIMARY}" opacity="0.85"/>
    <polygon points="-20,70 150,10 320,70" fill="{PRIMARY_DARK}" opacity="0.9"/>
    <rect x="130" y="30" width="40" height="40" fill="{banner_bg_1}"/>
    <rect x="20" y="100" width="26" height="80" fill="{banner_bg_1}"/>
    <rect x="60" y="100" width="26" height="45" fill="{banner_bg_1}" opacity="0.9"/>
    <rect x="100" y="100" width="26" height="45" fill="{banner_bg_1}" opacity="0.9"/>
    <rect x="174" y="100" width="26" height="45" fill="{banner_bg_1}" opacity="0.9"/>
    <rect x="214" y="100" width="26" height="45" fill="{banner_bg_1}" opacity="0.9"/>
    <rect x="254" y="100" width="26" height="80" fill="{banner_bg_1}"/>
    <rect x="0" y="176" width="300" height="6" fill="{PRIMARY_DARK}"/>
  </g>

  <!-- trees, kept simple and calm -->
  <g opacity="0.9">
    <circle cx="220" cy="185" r="26" fill="{PRIMARY}"/>
    <rect x="216" y="200" width="8" height="26" fill="{PRIMARY_DARK}"/>
    <circle cx="700" cy="190" r="30" fill="{PRIMARY}"/>
    <rect x="695" y="208" width="9" height="28" fill="{PRIMARY_DARK}"/>
    <circle cx="760" cy="180" r="20" fill="{ACCENT_WARM}" opacity="0.7"/>
    <rect x="757" y="194" width="6" height="22" fill="{PRIMARY_DARK}"/>
  </g>
</svg>
"""

campus_image_uri = _load_campus_image_data_uri()

if campus_image_uri:
    st.markdown(
        f"""
        <div class="campus-banner">
            <img src="{campus_image_uri}" alt="Keshav Mahavidyalaya campus">
            <div class="campus-banner-overlay">
                <h1>Keshav Mahavidyalaya PG Finder</h1>
                <p>Browse PGs near Pitampura and get a fair rent estimate.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="campus-banner" style="height:200px;">
            {CAMPUS_ILLUSTRATION_SVG}
            <div class="campus-banner-overlay">
                <h1>Keshav Mahavidyalaya PG Finder</h1>
                <p>Browse PGs near Pitampura and get a fair rent estimate.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MOCK PG DATABASE
# ============================================================
@st.cache_data
def load_pg_database() -> pd.DataFrame:
    data = [
        ("Shri Balaji PG", "Pitampura", 0.6, [1, 2], "Boys Only", 1, 1, 1, 1, 0, 4.3, 58, True, 2, 11000, "10:30 PM", "R. Sharma", "+91 98xxxx1234", 11000, 28.6985, 77.1320),
        ("Green View Residency", "Rani Bagh", 1.4, [2, 3], "Co-ed", 1, 0, 1, 0, 1, 3.9, 34, False, 0, 6000, "11:00 PM", "S. Verma", "+91 98xxxx5678", 7800, 28.6889, 77.1288),
        ("Pitampura Student Home", "Kohat Enclave", 2.1, [3], "Boys Only", 0, 0, 0, 0, 0, 3.5, 19, False, 4, 4000, "No Curfew", "M. Khan", "+91 98xxxx4321", 5500, 28.6928, 77.1372),
        ("Royal Girls Accommodation", "Pitampura", 0.9, [1], "Girls Only", 1, 1, 1, 1, 1, 4.6, 71, True, 1, 12000, "9:30 PM", "A. Gupta", "+91 98xxxx8765", 12500, 28.6992, 77.1305),
        ("Aggarwal PG", "Rani Bagh", 1.6, [2], "Girls Only", 1, 1, 0, 1, 0, 4.0, 27, True, 3, 8000, "10:00 PM", "N. Aggarwal", "+91 98xxxx1122", 8200, 28.6875, 77.1265),
        ("Metro Nest PG", "Pitampura", 0.4, [1, 2], "Co-ed", 0, 1, 1, 1, 1, 4.1, 45, True, 0, 12000, "11:30 PM", "V. Kapoor", "+91 98xxxx3344", 12000, 28.6970, 77.1300),
        ("Comfort Stay Boys PG", "Kohat Enclave", 2.4, [3], "Boys Only", 0, 0, 1, 0, 0, 3.7, 22, False, 5, 4500, "No Curfew", "D. Yadav", "+91 98xxxx5566", 5900, 28.6940, 77.1385),
        ("Sunrise Girls PG", "Wazirpur", 3.0, [1, 2], "Girls Only", 1, 1, 1, 1, 0, 4.2, 39, True, 2, 9500, "9:00 PM", "P. Chawla", "+91 98xxxx7788", 9800, 28.6820, 77.1580),
        ("City Comfort PG", "Wazirpur", 3.2, [2, 3], "Co-ed", 1, 0, 1, 0, 1, 3.6, 15, False, 3, 5500, "10:30 PM", "K. Malhotra", "+91 98xxxx9911", 7200, 28.6805, 77.1610),
        ("Elite Stay PG", "Pitampura", 0.7, [1], "Boys Only", 1, 1, 1, 1, 1, 4.5, 62, True, 1, 13000, "11:00 PM", "T. Bhatia", "+91 98xxxx2233", 13500, 28.6978, 77.1330),
        ("Happy Homes PG", "Rani Bagh", 1.2, [1, 2], "Girls Only", 1, 0, 1, 1, 0, 4.0, 30, False, 2, 7000, "10:00 PM", "S. Rani", "+91 98xxxx6644", 8600, 28.6895, 77.1275),
        ("Budget Boys Hostel", "Kohat Enclave", 2.6, [3], "Boys Only", 0, 0, 0, 0, 0, 3.2, 11, False, 6, 3500, "No Curfew", "L. Singh", "+91 98xxxx7722", 4800, 28.6950, 77.1400),
        ("Prime Living PG", "Pitampura", 1.0, [2], "Co-ed", 1, 1, 1, 1, 1, 4.4, 48, True, 1, 10000, "11:30 PM", "H. Arora", "+91 98xxxx8899", 10200, 28.6965, 77.1315),
        ("Vaishnavi Girls PG", "Rani Bagh", 1.8, [1, 2], "Girls Only", 1, 1, 0, 1, 0, 4.1, 24, True, 0, 8500, "9:30 PM", "M. Iyer", "+91 98xxxx3311", 8900, 28.6862, 77.1250),
        ("Student Nest Co-living", "Wazirpur", 2.9, [2, 3], "Co-ed", 1, 1, 1, 1, 1, 4.3, 33, True, 4, 9000, "No Curfew", "J. Thomas", "+91 98xxxx4488", 9600, 28.6835, 77.1560),
    ]
    cols = [
        "PG Name", "Area", "Distance (km)", "Sharing Options", "Gender", "Food Included",
        "AC", "WiFi", "Laundry", "Parking", "Rating", "Reviews Count", "Verified",
        "Rooms Available", "Security Deposit", "Curfew", "Owner", "Contact",
        "Monthly Rent (₹)", "Lat", "Lon",
    ]
    df = pd.DataFrame(data, columns=cols)
    df["Sharing Label"] = df["Sharing Options"].apply(
        lambda opts: " / ".join({1: "Single", 2: "Double", 3: "Triple"}[o] for o in opts)
    )
    return df


pg_database = load_pg_database()

SAMPLE_REVIEWS = [
    ("Ankit", 4, "Decent place, close to college. Rooms are clean and the owner is responsive."),
    ("Priya", 5, "Loved staying here — food is good and the area feels safe at night."),
    ("Rohan", 3, "Value for money but WiFi speed could be better during peak hours."),
]

# ============================================================
# ML MODEL (kept, but only the number is shown — no charts)
# ============================================================
@st.cache_resource
def train_rent_model():
    rng = np.random.default_rng(42)
    n = 260
    area_choices = np.array(["Pitampura", "Rani Bagh", "Kohat Enclave", "Wazirpur"])
    area_premium = {"Pitampura": 1500, "Rani Bagh": 400, "Kohat Enclave": -300, "Wazirpur": -600}

    distance = rng.uniform(0.2, 4.0, n)
    sharing = rng.choice([1, 2, 3], n, p=[0.35, 0.4, 0.25])
    ac = rng.choice([0, 1], n, p=[0.45, 0.55])
    food = rng.choice([0, 1], n, p=[0.4, 0.6])
    wifi = rng.choice([0, 1], n, p=[0.4, 0.6])
    laundry = rng.choice([0, 1], n, p=[0.55, 0.45])
    area = rng.choice(area_choices, n)

    base = 15000 - distance * 1800 - (sharing - 1) * 3200
    addons = ac * 1300 + food * 900 + wifi * 500 + laundry * 400
    premium = np.array([area_premium[a] for a in area])
    noise = rng.normal(0, 500, n)
    rent = np.clip(base + addons + premium + noise, 3500, None)

    df = pd.DataFrame(
        {
            "Distance": distance, "Sharing": sharing, "AC": ac, "Food": food,
            "WiFi": wifi, "Laundry": laundry, "Area": area, "Rent": rent,
        }
    )
    df = pd.get_dummies(df, columns=["Area"], prefix="Area")
    feature_cols = [c for c in df.columns if c != "Rent"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["Rent"], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, feature_cols, mae


rent_model, feature_cols, model_mae = train_rent_model()


def predict_rent(distance, sharing, ac, food, wifi, laundry, area):
    row = {c: 0 for c in feature_cols}
    row["Distance"] = distance
    row["Sharing"] = sharing
    row["AC"] = ac
    row["Food"] = food
    row["WiFi"] = wifi
    row["Laundry"] = laundry
    area_col = f"Area_{area}"
    if area_col in row:
        row[area_col] = 1
    X = pd.DataFrame([row])[feature_cols]
    return rent_model.predict(X)[0]


# ============================================================
# SIDEBAR — FILTERS
# ============================================================
st.sidebar.header("Filter PGs")

search_term = st.sidebar.text_input("Search by PG name")
selected_hub = st.sidebar.selectbox("Preferred Area", ["All Hubs"] + sorted(pg_database["Area"].unique()))
gender_policy = st.sidebar.selectbox("Gender Policy", ["All", "Boys Only", "Girls Only", "Co-ed"])
sharing_filter = st.sidebar.multiselect("Sharing Type", options=["Single", "Double", "Triple"], default=[])

col_a, col_b = st.sidebar.columns(2)
with col_a:
    food_incl = st.checkbox("Food", value=False)
    wifi_only = st.checkbox("WiFi", value=False)
with col_b:
    ac_only = st.checkbox("AC", value=False)
    laundry_only = st.checkbox("Laundry", value=False)

verified_only = st.sidebar.checkbox("Verified listings only", value=False)
available_only = st.sidebar.checkbox("Rooms currently available", value=False)

min_rent, max_rent = int(pg_database["Monthly Rent (₹)"].min()), int(pg_database["Monthly Rent (₹)"].max())
budget_range = st.sidebar.slider("Budget Range (₹ / month)", min_rent, max_rent, (min_rent, max_rent), step=100)
max_distance = st.sidebar.slider("Max distance from college (km)", 0.1, 4.0, 4.0, 0.1)
min_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)

sort_by = st.sidebar.selectbox(
    "Sort listings by",
    ["Distance (nearest first)", "Rent (lowest first)", "Rent (highest first)", "Rating (highest first)"],
)

# ============================================================
# FILTER LOGIC
# ============================================================
filtered_df = pg_database.copy()

if selected_hub != "All Hubs":
    filtered_df = filtered_df[filtered_df["Area"] == selected_hub]
if gender_policy != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender_policy]
if sharing_filter:
    label_map = {"Single": 1, "Double": 2, "Triple": 3}
    wanted = {label_map[s] for s in sharing_filter}
    filtered_df = filtered_df[filtered_df["Sharing Options"].apply(lambda opts: bool(wanted & set(opts)))]
if food_incl:
    filtered_df = filtered_df[filtered_df["Food Included"] == 1]
if wifi_only:
    filtered_df = filtered_df[filtered_df["WiFi"] == 1]
if ac_only:
    filtered_df = filtered_df[filtered_df["AC"] == 1]
if laundry_only:
    filtered_df = filtered_df[filtered_df["Laundry"] == 1]
if verified_only:
    filtered_df = filtered_df[filtered_df["Verified"]]
if available_only:
    filtered_df = filtered_df[filtered_df["Rooms Available"] > 0]

filtered_df = filtered_df[
    (filtered_df["Monthly Rent (₹)"] >= budget_range[0]) & (filtered_df["Monthly Rent (₹)"] <= budget_range[1])
]
filtered_df = filtered_df[filtered_df["Distance (km)"] <= max_distance]
filtered_df = filtered_df[filtered_df["Rating"] >= min_rating]

if search_term:
    filtered_df = filtered_df[filtered_df["PG Name"].str.contains(search_term, case=False)]

sort_map = {
    "Distance (nearest first)": ("Distance (km)", True),
    "Rent (lowest first)": ("Monthly Rent (₹)", True),
    "Rent (highest first)": ("Monthly Rent (₹)", False),
    "Rating (highest first)": ("Rating", False),
}
sort_col, ascending = sort_map[sort_by]
filtered_df = filtered_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

# ============================================================
# SESSION STATE
# ============================================================
if "favourites" not in st.session_state:
    st.session_state.favourites = set()
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []
if "page" not in st.session_state:
    st.session_state.page = 1

PAGE_SIZE = 5

# ============================================================
# TOP METRICS
# ============================================================
m1, m2, m3, m4 = st.columns(4)
metric_defs = [
    (m1, f"{len(filtered_df)}", "Matching PGs"),
    (m2, f"₹{int(filtered_df['Monthly Rent (₹)'].mean()):,}" if not filtered_df.empty else "—", "Avg. Rent"),
    (m3, f"{round(filtered_df['Distance (km)'].mean(), 1)} km" if not filtered_df.empty else "—", "Avg. Distance"),
    (m4, f"{len(st.session_state.favourites)}", "Favourites"),
]
for col, val, label in metric_defs:
    with col:
        st.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# MAIN TABS
# ============================================================
tab_estimator, tab_browse, tab_compare, tab_charts, tab_map, tab_fav, tab_faq = st.tabs(
    ["Rent Estimator", "Browse PGs", "Compare", "Charts", "Map", "Favourites", "FAQ"]
)

# Shared, calm chart styling so every figure in the "Charts" tab matches the app
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=text_color, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
GRID_COLOR = border_color

# ------------------------------------------------------------
# TAB — RENT ESTIMATOR
# ------------------------------------------------------------
with tab_estimator:
    st.subheader("Estimate a fair monthly rent")
    st.write("Enter the details below to get a rent estimate based on nearby listings.")

    col1, col2, col3 = st.columns(3)
    with col1:
        dist_input = st.slider("Distance from college (km)", 0.1, 4.0, 1.0, 0.1)
        area_input = st.selectbox("Area", sorted(pg_database["Area"].unique()))
    with col2:
        sharing_input = st.selectbox(
            "Sharing Type", options=[1, 2, 3],
            format_func=lambda x: {1: "Single", 2: "Double", 3: "Triple"}[x],
        )
        ac_input = st.radio("AC?", ["Yes", "No"], horizontal=True)
    with col3:
        food_input = st.radio("Food?", ["Yes", "No"], horizontal=True)
        wifi_input = st.radio("WiFi?", ["Yes", "No"], horizontal=True)
    laundry_input = st.checkbox("Laundry service included", value=False)

    if st.button("Estimate rent", type="primary"):
        pred = predict_rent(
            dist_input, sharing_input,
            1 if ac_input == "Yes" else 0,
            1 if food_input == "Yes" else 0,
            1 if wifi_input == "Yes" else 0,
            1 if laundry_input else 0,
            area_input,
        )
        margin = max(model_mae, 400)
        st.success(f"Estimated fair rent: ₹{int(pred):,} / month")
        st.caption(f"Likely range: ₹{int(pred - margin):,} – ₹{int(pred + margin):,} / month")

        similar = pg_database[pg_database["Sharing Options"].apply(lambda opts: sharing_input in opts)]
        if not similar.empty:
            st.markdown("**Similar listings nearby:**")
            st.table(
                similar[["PG Name", "Area", "Monthly Rent (₹)"]]
                .sort_values("Monthly Rent (₹)")
                .reset_index(drop=True)
            )

        with st.expander("Full cost breakdown (rent + deposit)"):
            avg_deposit = int(similar["Security Deposit"].mean()) if not similar.empty else 8000
            months = st.slider("Spread the deposit over how many months?", 3, 24, 11, key="amort")
            monthly_effective = pred + avg_deposit / months
            st.write(f"Typical security deposit for this sharing type: ₹{avg_deposit:,}")
            st.write(f"Effective monthly cost including the deposit: ₹{int(monthly_effective):,}")

    with st.expander("How is this estimate calculated?"):
        st.write(
            "The estimate is based on a model trained on sample PG pricing patterns near the "
            "college, factoring in distance, sharing type, area and amenities. It's meant as a "
            "starting reference — always confirm the final price with the PG owner."
        )

# ------------------------------------------------------------
# TAB — BROWSE PGS
# ------------------------------------------------------------
with tab_browse:
    st.subheader("Available listings near college")

    if filtered_df.empty:
        st.warning("No PGs match your current filters. Try widening your budget or removing a filter.")
    else:
        csv = filtered_df.drop(columns=["Lat", "Lon", "Sharing Options"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", data=csv, file_name="km_pg_listings.csv", mime="text/csv")

        total_pages = max(1, math.ceil(len(filtered_df) / PAGE_SIZE))
        st.session_state.page = min(st.session_state.page, total_pages)
        start = (st.session_state.page - 1) * PAGE_SIZE
        page_df = filtered_df.iloc[start:start + PAGE_SIZE]

        for _, row in page_df.iterrows():
            is_fav = row["PG Name"] in st.session_state.favourites
            in_compare = row["PG Name"] in st.session_state.compare_list

            with st.container():
                st.markdown('<div class="pg-card">', unsafe_allow_html=True)
                info_col, action_col = st.columns([4, 1.1])

                with info_col:
                    verified_badge = '<span class="chip chip-verified">Verified</span>' if row["Verified"] else ""
                    low_rooms_badge = (
                        '<span class="chip chip-lowrooms">Filling fast</span>'
                        if 0 < row["Rooms Available"] <= 1 else ""
                    )
                    sold_out = row["Rooms Available"] == 0
                    st.markdown(f"#### {row['PG Name']}  ·  {row['Rating']} rating ({row['Reviews Count']} reviews)")
                    st.markdown(
                        f'<span class="chip">{row["Area"]}</span>'
                        f'<span class="chip">{row["Gender"]}</span>'
                        f'<span class="chip">{row["Sharing Label"]}</span>'
                        f'<span class="chip">{row["Distance (km)"]} km away</span>'
                        f'{verified_badge}{low_rooms_badge}',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<span class="rent-tag">₹{row["Monthly Rent (₹)"]:,} / month</span>'
                                f' &nbsp; <span style="font-size:0.85rem;">+ ₹{row["Security Deposit"]:,} deposit</span>',
                                unsafe_allow_html=True)
                    amenities = []
                    if row["AC"]:
                        amenities.append("AC")
                    if row["Food Included"]:
                        amenities.append("Food")
                    if row["WiFi"]:
                        amenities.append("WiFi")
                    if row["Laundry"]:
                        amenities.append("Laundry")
                    if row["Parking"]:
                        amenities.append("Parking")
                    st.caption((", ".join(amenities) if amenities else "Basic amenities")
                               + f"  ·  Curfew: {row['Curfew']}")
                    if sold_out:
                        st.caption("No rooms currently available")
                    else:
                        st.caption(f"{row['Rooms Available']} room(s) available")

                    with st.expander("Reviews"):
                        for reviewer, stars, text in SAMPLE_REVIEWS:
                            st.markdown(
                                f'<div class="review-box"><b>{reviewer}</b> — {stars}/5<br>{text}</div>',
                                unsafe_allow_html=True,
                            )

                with action_col:
                    fav_label = "Unsave" if is_fav else "Save"
                    if st.button(fav_label, key=f"fav_{row['PG Name']}"):
                        if is_fav:
                            st.session_state.favourites.discard(row["PG Name"])
                        else:
                            st.session_state.favourites.add(row["PG Name"])
                        st.rerun()

                    compare_label = "Remove" if in_compare else "Compare"
                    disabled_compare = (not in_compare) and len(st.session_state.compare_list) >= 3
                    if st.button(compare_label, key=f"cmp_{row['PG Name']}", disabled=disabled_compare):
                        if in_compare:
                            st.session_state.compare_list.remove(row["PG Name"])
                        else:
                            st.session_state.compare_list.append(row["PG Name"])
                        st.rerun()

                    with st.expander("Enquire"):
                        with st.form(key=f"enquire_{row['PG Name']}"):
                            st.text_input("Your name", key=f"name_{row['PG Name']}")
                            st.text_input("Your phone", key=f"phone_{row['PG Name']}")
                            st.text_area("Message", value=f"Hi, I'm interested in {row['PG Name']}.",
                                         key=f"msg_{row['PG Name']}")
                            submitted = st.form_submit_button("Send enquiry")
                            if submitted:
                                st.success(f"Noted. {row['Owner']} can be reached at {row['Contact']} "
                                           f"(demo only — no message is actually sent).")

                st.markdown("</div>", unsafe_allow_html=True)

        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("Previous", disabled=st.session_state.page <= 1):
                st.session_state.page -= 1
                st.rerun()
        with nav2:
            st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} of {total_pages}</div>",
                        unsafe_allow_html=True)
        with nav3:
            if st.button("Next", disabled=st.session_state.page >= total_pages):
                st.session_state.page += 1
                st.rerun()

# ------------------------------------------------------------
# TAB — COMPARE
# ------------------------------------------------------------
with tab_compare:
    st.subheader("Side-by-side comparison")
    if not st.session_state.compare_list:
        st.info("Add up to 3 PGs to compare using the 'Compare' button in 'Browse PGs'.")
    else:
        compare_df = pg_database[pg_database["PG Name"].isin(st.session_state.compare_list)]
        display_cols = [
            "PG Name", "Area", "Distance (km)", "Sharing Label", "Gender", "Monthly Rent (₹)",
            "Security Deposit", "Rating", "AC", "Food Included", "WiFi", "Laundry", "Verified", "Curfew",
        ]
        st.dataframe(compare_df[display_cols].set_index("PG Name").T, use_container_width=True)
        if st.button("Clear comparison"):
            st.session_state.compare_list = []
            st.rerun()

# ------------------------------------------------------------
# TAB — CHARTS
# ------------------------------------------------------------
with tab_charts:
    st.subheader("Compare PGs at a glance")
    st.markdown(
        '<p class="section-note">These charts reflect your current filters from the sidebar, '
        "so they update as you narrow down your search.</p>",
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning("No PGs match your current filters, so there's nothing to chart yet. Try widening your search.")
    else:
        chart_df = filtered_df.sort_values("Monthly Rent (₹)")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Monthly rent by PG**")
            fig_rent = px.bar(
                chart_df,
                x="Monthly Rent (₹)",
                y="PG Name",
                orientation="h",
                color="Area",
                color_discrete_sequence=CHART_PALETTE,
            )
            fig_rent.update_layout(**PLOTLY_LAYOUT, height=max(320, 34 * len(chart_df)), showlegend=True)
            fig_rent.update_xaxes(gridcolor=GRID_COLOR, title="₹ / month")
            fig_rent.update_yaxes(title="", gridcolor=GRID_COLOR)
            st.plotly_chart(fig_rent, use_container_width=True)

        with c2:
            st.markdown("**Rent vs. distance from college**")
            fig_scatter = px.scatter(
                filtered_df,
                x="Distance (km)",
                y="Monthly Rent (₹)",
                color="Area",
                size="Rating",
                hover_name="PG Name",
                color_discrete_sequence=CHART_PALETTE,
            )
            fig_scatter.update_layout(**PLOTLY_LAYOUT, height=max(320, 34 * len(chart_df)))
            fig_scatter.update_xaxes(gridcolor=GRID_COLOR, title="Distance (km)")
            fig_scatter.update_yaxes(gridcolor=GRID_COLOR, title="₹ / month")
            st.plotly_chart(fig_scatter, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**Average rent by area**")
            area_avg = filtered_df.groupby("Area", as_index=False)["Monthly Rent (₹)"].mean()
            fig_area = px.bar(
                area_avg.sort_values("Monthly Rent (₹)"),
                x="Area",
                y="Monthly Rent (₹)",
                color="Area",
                color_discrete_sequence=CHART_PALETTE,
            )
            fig_area.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
            fig_area.update_xaxes(title="")
            fig_area.update_yaxes(gridcolor=GRID_COLOR, title="Avg. ₹ / month")
            st.plotly_chart(fig_area, use_container_width=True)

        with c4:
            st.markdown("**How well-reviewed are these PGs?**")
            fig_rating = px.bar(
                chart_df.sort_values("Rating"),
                x="Rating",
                y="PG Name",
                orientation="h",
                color="Verified",
                color_discrete_map={True: PRIMARY, False: ACCENT_WARM},
            )
            fig_rating.update_layout(**PLOTLY_LAYOUT, height=max(320, 34 * len(chart_df)))
            fig_rating.update_xaxes(gridcolor=GRID_COLOR, title="Rating (out of 5)", range=[0, 5])
            fig_rating.update_yaxes(title="")
            st.plotly_chart(fig_rating, use_container_width=True)

        st.markdown("**Amenities available across matching PGs**")
        amenity_cols = ["Food Included", "AC", "WiFi", "Laundry", "Parking"]
        amenity_share = (
            filtered_df[amenity_cols].mean().mul(100).round(0).reset_index()
        )
        amenity_share.columns = ["Amenity", "Share of listings (%)"]
        fig_amenities = px.bar(
            amenity_share.sort_values("Share of listings (%)"),
            x="Share of listings (%)",
            y="Amenity",
            orientation="h",
            color_discrete_sequence=[PRIMARY],
        )
        fig_amenities.update_traces(marker_color=PRIMARY)
        fig_amenities.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        fig_amenities.update_xaxes(gridcolor=GRID_COLOR, title="% of matching PGs", range=[0, 100])
        fig_amenities.update_yaxes(title="")
        st.plotly_chart(fig_amenities, use_container_width=True)

        if st.session_state.compare_list:
            st.divider()
            st.markdown("**Your shortlist, side by side**")
            shortlist_df = pg_database[pg_database["PG Name"].isin(st.session_state.compare_list)]
            fig_shortlist = go.Figure()
            fig_shortlist.add_trace(go.Bar(
                name="Monthly Rent (₹)", x=shortlist_df["PG Name"], y=shortlist_df["Monthly Rent (₹)"],
                marker_color=PRIMARY,
            ))
            fig_shortlist.add_trace(go.Bar(
                name="Security Deposit (₹)", x=shortlist_df["PG Name"], y=shortlist_df["Security Deposit"],
                marker_color=ACCENT_WARM,
            ))
            fig_shortlist.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340)
            fig_shortlist.update_xaxes(title="")
            fig_shortlist.update_yaxes(gridcolor=GRID_COLOR, title="₹")
            st.plotly_chart(fig_shortlist, use_container_width=True)
        else:
            st.caption("Tip: add PGs to your shortlist from 'Browse PGs' to see a rent-vs-deposit comparison here.")

# ------------------------------------------------------------
# TAB — MAP
# ------------------------------------------------------------
with tab_map:
    st.subheader("Where these PGs are, roughly")
    if filtered_df.empty:
        st.info("No listings to show on the map — adjust your filters.")
    else:
        st.map(filtered_df.rename(columns={"Lat": "lat", "Lon": "lon"})[["lat", "lon"]], size=40)
        st.caption("Pin locations are approximate and for general orientation only.")

# ------------------------------------------------------------
# TAB — FAVOURITES
# ------------------------------------------------------------
with tab_fav:
    st.subheader("Your saved PGs")
    fav_df = pg_database[pg_database["PG Name"].isin(st.session_state.favourites)]
    if fav_df.empty:
        st.info("You haven't saved any PGs yet. Go to 'Browse PGs' and tap Save on a listing.")
    else:
        st.dataframe(
            fav_df.drop(columns=["Lat", "Lon", "Sharing Options"]),
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------------
# TAB — FAQ
# ------------------------------------------------------------
with tab_faq:
    st.subheader("Frequently asked questions")
    faqs = [
        ("Is the rent estimate exact?", "No — it's a data-driven guide based on distance, sharing type and "
         "amenities. Always confirm the final price with the PG owner."),
        ("How do I save a PG for later?", "Tap Save on any listing in the 'Browse PGs' tab; find it later "
         "under the 'Favourites' tab."),
        ("Can I compare more than 3 PGs?", "The comparison table is capped at 3 for readability — remove one "
         "before adding another."),
        ("Is this real-time data?", "No, listings and pricing here are illustrative sample data for demo "
         "purposes, not live inventory."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.write(a)

st.divider()
st.caption(
    "Built for Keshav Mahavidyalaya students. Listings, reviews and rent estimates in this demo are "
    "illustrative sample data, not verified real-time information. Always confirm details directly with "
    "PG owners before paying any deposit."
)