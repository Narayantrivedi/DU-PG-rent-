"""
Keshav Mahavidyalaya PG Smart Finder & Rent Predictor  —  v2
--------------------------------------------------------------
A richer Streamlit app to help students find PG (paying-guest) accommodation
around Pitampura / Keshav Mahavidyalaya, compare options side-by-side, and
get an ML-estimated fair monthly rent.

Run with:  streamlit run pg_finder_app.py
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KM PG Finder & Rent Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME TOGGLE (light / dark accent)
# ============================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    bg, card_bg, text_color, sub_text = "#0f1117", "#1b1e29", "#f1f1f5", "#a8adba"
else:
    bg, card_bg, text_color, sub_text = "#f7f9fc", "#ffffff", "#1c1c26", "#5b5f6d"

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background: {bg};
        color: {text_color};
    }}

    .hero {{
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #4b3f72 0%, #7c5cbf 45%, #b98dd6 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(75, 63, 114, 0.25);
    }}
    .hero h1 {{ margin-bottom: 0.2rem; font-size: 1.8rem; }}
    .hero p {{ margin: 0; opacity: 0.9; font-size: 0.95rem; }}

    .metric-card {{
        background: {card_bg};
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        text-align: center;
        color: {text_color};
    }}
    .metric-card h3 {{ margin: 0; }}
    .metric-card p {{ margin: 0; color: {sub_text}; font-size: 0.82rem; }}

    .pg-card {{
        background: {card_bg};
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.10);
        border-left: 5px solid #7c5cbf;
        color: {text_color};
    }}
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-bottom: 4px;
        background: #f0ecfa;
        color: #4b3f72;
        font-weight: 600;
    }}
    .badge-verified {{ background: #e3f7ea; color: #1e7d43; }}
    .badge-lowrooms {{ background: #fdeaea; color: #b3261e; }}
    .rent-tag {{ font-size: 1.2rem; font-weight: 700; color: #2f6f4f; }}
    .review-box {{
        background: rgba(124, 92, 191, 0.08);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.88rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Keshav Mahavidyalaya PG Smart Finder & Rent Predictor</h1>
        <p>Browse PGs around Pitampura, compare them side-by-side, and get an
        ML-estimated fair monthly rent — built for Keshav Mahavidyalaya students.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MOCK PG DATABASE (larger + richer)
# ============================================================
@st.cache_data
def load_pg_database() -> pd.DataFrame:
    data = [
        # name, area, distance, sharing_options, gender, food, ac, wifi, laundry, parking,
        # rating, reviews_count, verified, rooms_available, deposit, curfew, owner, contact, rent, lat, lon, img_seed
        ("Shri Balaji PG", "Pitampura", 0.6, [1, 2], "Boys Only", 1, 1, 1, 1, 0, 4.3, 58, True, 2, 11000, "10:30 PM", "R. Sharma", "+91 98xxxx1234", 11000, 28.6985, 77.1320, 21),
        ("Green View Residency", "Rani Bagh", 1.4, [2, 3], "Co-ed", 1, 0, 1, 0, 1, 3.9, 34, False, 0, 6000, "11:00 PM", "S. Verma", "+91 98xxxx5678", 7800, 28.6889, 77.1288, 22),
        ("Pitampura Student Home", "Kohat Enclave", 2.1, [3], "Boys Only", 0, 0, 0, 0, 0, 3.5, 19, False, 4, 4000, "No Curfew", "M. Khan", "+91 98xxxx4321", 5500, 28.6928, 77.1372, 23),
        ("Royal Girls Accommodation", "Pitampura", 0.9, [1], "Girls Only", 1, 1, 1, 1, 1, 4.6, 71, True, 1, 12000, "9:30 PM", "A. Gupta", "+91 98xxxx8765", 12500, 28.6992, 77.1305, 24),
        ("Aggarwal PG", "Rani Bagh", 1.6, [2], "Girls Only", 1, 1, 0, 1, 0, 4.0, 27, True, 3, 8000, "10:00 PM", "N. Aggarwal", "+91 98xxxx1122", 8200, 28.6875, 77.1265, 25),
        ("Metro Nest PG", "Pitampura", 0.4, [1, 2], "Co-ed", 0, 1, 1, 1, 1, 4.1, 45, True, 0, 12000, "11:30 PM", "V. Kapoor", "+91 98xxxx3344", 12000, 28.6970, 77.1300, 26),
        ("Comfort Stay Boys PG", "Kohat Enclave", 2.4, [3], "Boys Only", 0, 0, 1, 0, 0, 3.7, 22, False, 5, 4500, "No Curfew", "D. Yadav", "+91 98xxxx5566", 5900, 28.6940, 77.1385, 27),
        ("Sunrise Girls PG", "Wazirpur", 3.0, [1, 2], "Girls Only", 1, 1, 1, 1, 0, 4.2, 39, True, 2, 9500, "9:00 PM", "P. Chawla", "+91 98xxxx7788", 9800, 28.6820, 77.1580, 28),
        ("City Comfort PG", "Wazirpur", 3.2, [2, 3], "Co-ed", 1, 0, 1, 0, 1, 3.6, 15, False, 3, 5500, "10:30 PM", "K. Malhotra", "+91 98xxxx9911", 7200, 28.6805, 77.1610, 29),
        ("Elite Stay PG", "Pitampura", 0.7, [1], "Boys Only", 1, 1, 1, 1, 1, 4.5, 62, True, 1, 13000, "11:00 PM", "T. Bhatia", "+91 98xxxx2233", 13500, 28.6978, 77.1330, 30),
        ("Happy Homes PG", "Rani Bagh", 1.2, [1, 2], "Girls Only", 1, 0, 1, 1, 0, 4.0, 30, False, 2, 7000, "10:00 PM", "S. Rani", "+91 98xxxx6644", 8600, 28.6895, 77.1275, 31),
        ("Budget Boys Hostel", "Kohat Enclave", 2.6, [3], "Boys Only", 0, 0, 0, 0, 0, 3.2, 11, False, 6, 3500, "No Curfew", "L. Singh", "+91 98xxxx7722", 4800, 28.6950, 77.1400, 32),
        ("Prime Living PG", "Pitampura", 1.0, [2], "Co-ed", 1, 1, 1, 1, 1, 4.4, 48, True, 1, 10000, "11:30 PM", "H. Arora", "+91 98xxxx8899", 10200, 28.6965, 77.1315, 33),
        ("Vaishnavi Girls PG", "Rani Bagh", 1.8, [1, 2], "Girls Only", 1, 1, 0, 1, 0, 4.1, 24, True, 0, 8500, "9:30 PM", "M. Iyer", "+91 98xxxx3311", 8900, 28.6862, 77.1250, 34),
        ("Student Nest Co-living", "Wazirpur", 2.9, [2, 3], "Co-ed", 1, 1, 1, 1, 1, 4.3, 33, True, 4, 9000, "No Curfew", "J. Thomas", "+91 98xxxx4488", 9600, 28.6835, 77.1560, 35),
    ]
    cols = [
        "PG Name", "Area", "Distance (km)", "Sharing Options", "Gender", "Food Included",
        "AC", "WiFi", "Laundry", "Parking", "Rating", "Reviews Count", "Verified",
        "Rooms Available", "Security Deposit", "Curfew", "Owner", "Contact",
        "Monthly Rent (₹)", "Lat", "Lon", "Img Seed",
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
# ML MODEL — RandomForest on a larger synthetic dataset
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
    preds = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, preds), "mae": mean_absolute_error(y_test, preds)}
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return model, feature_cols, metrics, importances


rent_model, feature_cols, model_metrics, feature_importances = train_rent_model()


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
st.sidebar.header("🔍 Filter Nearby PGs")

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
m1, m2, m3, m4, m5 = st.columns(5)
metric_defs = [
    (m1, f"{len(filtered_df)}", "Matching PGs"),
    (m2, f"₹{int(filtered_df['Monthly Rent (₹)'].mean()):,}" if not filtered_df.empty else "—", "Avg. Rent"),
    (m3, f"{round(filtered_df['Distance (km)'].mean(), 1)} km" if not filtered_df.empty else "—", "Avg. Distance"),
    (m4, f"{round(filtered_df['Rating'].mean(), 1)} ⭐" if not filtered_df.empty else "—", "Avg. Rating"),
    (m5, f"{len(st.session_state.favourites)}", "Favourites"),
]
for col, val, label in metric_defs:
    with col:
        st.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# MAIN TABS
# ============================================================
tab_overview, tab_estimator, tab_browse, tab_compare, tab_map, tab_fav, tab_faq = st.tabs(
    ["📊 Dashboard", "💡 AI Rent Estimator", "🏠 Browse PGs", "⚖️ Compare", "🗺️ Map", "⭐ Favourites", "❓ FAQ"]
)

# ------------------------------------------------------------
# TAB — DASHBOARD
# ------------------------------------------------------------
with tab_overview:
    st.subheader("Market snapshot around Keshav Mahavidyalaya")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average rent by area**")
        st.bar_chart(pg_database.groupby("Area")["Monthly Rent (₹)"].mean())
    with c2:
        st.markdown("**Listings by gender policy**")
        st.bar_chart(pg_database["Gender"].value_counts())

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Rent vs. distance from college**")
        st.scatter_chart(pg_database, x="Distance (km)", y="Monthly Rent (₹)", color="Area")
    with c4:
        st.markdown("**What drives the rent estimate?**")
        st.bar_chart(feature_importances)
        st.caption("Feature importance from the Random Forest model (illustrative, trained on synthetic data).")

    st.info(
        f"Model performance on held-out synthetic data — R² ≈ {model_metrics['r2']:.2f}, "
        f"mean absolute error ≈ ₹{int(model_metrics['mae']):,}."
    )

# ------------------------------------------------------------
# TAB — AI RENT ESTIMATOR
# ------------------------------------------------------------
with tab_estimator:
    st.subheader("Machine Learning Rent Predictor")
    st.write("Now powered by a Random Forest model trained on a larger simulated dataset "
             "with area, amenities, sharing type and distance as inputs.")

    col1, col2, col3 = st.columns(3)
    with col1:
        dist_input = st.slider("Distance from college (km)", 0.1, 4.0, 1.0, 0.1)
        area_input = st.selectbox("Area", sorted(pg_database["Area"].unique()))
    with col2:
        sharing_input = st.selectbox(
            "Sharing Type", options=[1, 2, 3],
            format_func=lambda x: {1: "Single (1)", 2: "Double (2)", 3: "Triple (3)"}[x],
        )
        ac_input = st.radio("AC?", ["Yes", "No"], horizontal=True)
    with col3:
        food_input = st.radio("Food?", ["Yes", "No"], horizontal=True)
        wifi_input = st.radio("WiFi?", ["Yes", "No"], horizontal=True)
    laundry_input = st.checkbox("Laundry service included", value=False)

    if st.button("Predict Fair Rent", type="primary"):
        pred = predict_rent(
            dist_input, sharing_input,
            1 if ac_input == "Yes" else 0,
            1 if food_input == "Yes" else 0,
            1 if wifi_input == "Yes" else 0,
            1 if laundry_input else 0,
            area_input,
        )
        margin = max(model_metrics["mae"], 400)
        st.success(f"Estimated Fair Rent: **₹{int(pred):,} / month**")
        st.caption(f"Likely range: ₹{int(pred - margin):,} – ₹{int(pred + margin):,} / month "
                   f"(based on model error of ≈ ₹{int(model_metrics['mae']):,})")

        similar = pg_database[pg_database["Sharing Options"].apply(lambda opts: sharing_input in opts)]
        if not similar.empty:
            st.markdown("**How this compares to similar real listings:**")
            compare_chart = similar[["PG Name", "Monthly Rent (₹)"]].set_index("PG Name")
            compare_chart.loc["Your Estimate"] = int(pred)
            st.bar_chart(compare_chart)

        with st.expander("💰 Full cost breakdown (rent + deposit, amortised)"):
            avg_deposit = int(similar["Security Deposit"].mean()) if not similar.empty else 8000
            months = st.slider("Amortise deposit over how many months of stay?", 3, 24, 11, key="amort")
            monthly_effective = pred + avg_deposit / months
            st.write(f"Typical security deposit for this sharing type: **₹{avg_deposit:,}**")
            st.write(f"Effective monthly cost including amortised deposit: **₹{int(monthly_effective):,}**")

    with st.expander("ℹ️ How is this estimate calculated?"):
        st.write(
            "A Random Forest Regressor is trained on a simulated dataset of 260 PG profiles "
            "varying distance, sharing type, amenities and area. It captures non-linear effects "
            "(e.g. diminishing rent drop-off with distance) better than a plain linear model. "
            "This is illustrative — always confirm final pricing with the PG owner directly."
        )

# ------------------------------------------------------------
# TAB — BROWSE PGS
# ------------------------------------------------------------
with tab_browse:
    st.subheader("Available Listings Near College")

    if filtered_df.empty:
        st.warning("No PGs match your current filters. Try widening your budget or removing a filter.")
    else:
        csv = filtered_df.drop(columns=["Lat", "Lon", "Img Seed", "Sharing Options"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", data=csv, file_name="km_pg_listings.csv", mime="text/csv")

        total_pages = max(1, math.ceil(len(filtered_df) / PAGE_SIZE))
        st.session_state.page = min(st.session_state.page, total_pages)
        start = (st.session_state.page - 1) * PAGE_SIZE
        page_df = filtered_df.iloc[start:start + PAGE_SIZE]

        for _, row in page_df.iterrows():
            is_fav = row["PG Name"] in st.session_state.favourites
            in_compare = row["PG Name"] in st.session_state.compare_list

            with st.container():
                st.markdown('<div class="pg-card">', unsafe_allow_html=True)
                img_col, info_col, action_col = st.columns([1.1, 3.3, 1.1])

                with img_col:
                    st.image(f"https://picsum.photos/seed/{row['Img Seed']}/300/200", use_container_width=True)

                with info_col:
                    verified_badge = '<span class="badge badge-verified">✔ Verified</span>' if row["Verified"] else ""
                    low_rooms_badge = (
                        '<span class="badge badge-lowrooms">Filling fast</span>'
                        if 0 < row["Rooms Available"] <= 1 else ""
                    )
                    sold_out = row["Rooms Available"] == 0
                    st.markdown(f"#### {row['PG Name']}  ·  ⭐ {row['Rating']} ({row['Reviews Count']} reviews)")
                    st.markdown(
                        f'<span class="badge">{row["Area"]}</span>'
                        f'<span class="badge">{row["Gender"]}</span>'
                        f'<span class="badge">{row["Sharing Label"]}</span>'
                        f'<span class="badge">{row["Distance (km)"]} km away</span>'
                        f'{verified_badge}{low_rooms_badge}',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<span class="rent-tag">₹{row["Monthly Rent (₹)"]:,} / month</span>'
                                f' &nbsp; <span style="font-size:0.85rem;">+ ₹{row["Security Deposit"]:,} deposit</span>',
                                unsafe_allow_html=True)
                    amenities = []
                    if row["AC"]:
                        amenities.append("❄️ AC")
                    if row["Food Included"]:
                        amenities.append("🍽️ Food")
                    if row["WiFi"]:
                        amenities.append("📶 WiFi")
                    if row["Laundry"]:
                        amenities.append("🧺 Laundry")
                    if row["Parking"]:
                        amenities.append("🅿️ Parking")
                    st.caption((" · ".join(amenities) if amenities else "Basic amenities")
                               + f"  ·  Curfew: {row['Curfew']}")
                    if sold_out:
                        st.caption("🔴 No rooms currently available")
                    else:
                        st.caption(f"🟢 {row['Rooms Available']} room(s) available")

                    with st.expander("📖 Reviews"):
                        for reviewer, stars, text in SAMPLE_REVIEWS:
                            st.markdown(
                                f'<div class="review-box"><b>{reviewer}</b> {"⭐" * stars}<br>{text}</div>',
                                unsafe_allow_html=True,
                            )

                with action_col:
                    fav_label = "💔 Unsave" if is_fav else "❤️ Save"
                    if st.button(fav_label, key=f"fav_{row['PG Name']}"):
                        if is_fav:
                            st.session_state.favourites.discard(row["PG Name"])
                        else:
                            st.session_state.favourites.add(row["PG Name"])
                        st.rerun()

                    compare_label = "➖ Remove" if in_compare else "➕ Compare"
                    disabled_compare = (not in_compare) and len(st.session_state.compare_list) >= 3
                    if st.button(compare_label, key=f"cmp_{row['PG Name']}", disabled=disabled_compare):
                        if in_compare:
                            st.session_state.compare_list.remove(row["PG Name"])
                        else:
                            st.session_state.compare_list.append(row["PG Name"])
                        st.rerun()

                    with st.expander("✉️ Enquire"):
                        with st.form(key=f"enquire_{row['PG Name']}"):
                            name = st.text_input("Your name", key=f"name_{row['PG Name']}")
                            phone = st.text_input("Your phone", key=f"phone_{row['PG Name']}")
                            msg = st.text_area("Message", value=f"Hi, I'm interested in {row['PG Name']}.",
                                                key=f"msg_{row['PG Name']}")
                            submitted = st.form_submit_button("Send enquiry")
                            if submitted:
                                st.success(f"Enquiry noted! {row['Owner']} will be notified at {row['Contact']} "
                                           f"(demo only — no message is actually sent).")

                st.markdown("</div>", unsafe_allow_html=True)

        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("⬅️ Previous", disabled=st.session_state.page <= 1):
                st.session_state.page -= 1
                st.rerun()
        with nav2:
            st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} of {total_pages}</div>",
                        unsafe_allow_html=True)
        with nav3:
            if st.button("Next ➡️", disabled=st.session_state.page >= total_pages):
                st.session_state.page += 1
                st.rerun()

# ------------------------------------------------------------
# TAB — COMPARE
# ------------------------------------------------------------
with tab_compare:
    st.subheader("Side-by-side comparison")
    if not st.session_state.compare_list:
        st.info("Add up to 3 PGs to compare using the ➕ Compare button in 'Browse PGs'.")
    else:
        compare_df = pg_database[pg_database["PG Name"].isin(st.session_state.compare_list)]
        display_cols = [
            "PG Name", "Area", "Distance (km)", "Sharing Label", "Gender", "Monthly Rent (₹)",
            "Security Deposit", "Rating", "AC", "Food Included", "WiFi", "Laundry", "Verified", "Curfew",
        ]
        st.dataframe(
            compare_df[display_cols].set_index("PG Name").T,
            use_container_width=True,
        )
        if st.button("Clear comparison"):
            st.session_state.compare_list = []
            st.rerun()

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
    st.subheader("Your Saved PGs")
    fav_df = pg_database[pg_database["PG Name"].isin(st.session_state.favourites)]
    if fav_df.empty:
        st.info("You haven't saved any PGs yet. Go to 'Browse PGs' and tap ❤️ Save on a listing.")
    else:
        st.dataframe(
            fav_df.drop(columns=["Lat", "Lon", "Img Seed", "Sharing Options"]),
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
        ("How do I save a PG for later?", "Tap ❤️ Save on any listing in the 'Browse PGs' tab; find it later "
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
    "Built for Keshav Mahavidyalaya students · Listings, reviews and rent estimates in this demo are "
    "illustrative sample data, not verified real-time information. Always confirm details directly with "
    "PG owners before paying any deposit."
)