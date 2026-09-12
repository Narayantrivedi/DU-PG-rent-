"""
Keshav Mahavidyalaya PG Finder & Rent Estimator
--------------------------------------------------
A calm, Android-style Streamlit app to browse PG (paying-guest) listings
around Pitampura / Keshav Mahavidyalaya and estimate a fair monthly rent.

Run with:  streamlit run pg_finder_app.py
"""

import base64
import io
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
    page_title="KM PG Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CAMPUS PHOTO (embedded as base64 so this file works standalone —
# this is the photo of Keshav Mahavidyalaya you provided)
# ============================================================
CAMPUS_PHOTO_B64 = (
    "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"
)

# ============================================================
# THEME
# ============================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    bg, card_bg, text_color, sub_text, border_color = "#121212", "#1e1e1e", "#e8e8e8", "#a0a0a0", "#2c2c2c"
else:
    bg, card_bg, text_color, sub_text, border_color = "#fafafa", "#ffffff", "#212121", "#6b6b6b", "#e0e0e0"

PRIMARY = "#00796b"  # calm teal, Material-style accent

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    html, body, [class*="css"] {{
        font-family: 'Roboto', -apple-system, 'Segoe UI', sans-serif;
    }}

    .stApp {{
        background: {bg};
        color: {text_color};
    }}

    .page-title {{
        padding: 0.2rem 0 0.4rem 0;
        margin-bottom: 0;
    }}
    .page-title h1 {{
        font-size: 1.5rem;
        font-weight: 500;
        margin: 0;
        color: {text_color};
    }}
    .page-title p {{
        margin: 0.2rem 0 0.6rem 0;
        color: {sub_text};
        font-size: 0.92rem;
    }}
    .header-wrap {{
        border-bottom: 1px solid {border_color};
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
    }}

    .metric-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }}
    .metric-card h3 {{ margin: 0; font-weight: 500; color: {text_color}; }}
    .metric-card p {{ margin: 0; color: {sub_text}; font-size: 0.8rem; }}

    .pg-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        color: {text_color};
    }}
    .chip {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-bottom: 4px;
        background: {border_color};
        color: {text_color};
        font-weight: 500;
    }}
    .chip-verified {{ background: #dff0ea; color: #00695c; }}
    .chip-lowrooms {{ background: #fbe9e7; color: #b71c1c; }}
    .rent-tag {{ font-size: 1.1rem; font-weight: 500; color: {PRIMARY}; }}
    .review-box {{
        background: {bg};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.86rem;
    }}

    .stButton > button {{
        border-radius: 8px;
        font-family: 'Roboto', sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER (with embedded campus photo)
# ============================================================
st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
header_col1, header_col2 = st.columns([1, 2.4])

with header_col1:
    campus_bytes = base64.b64decode(CAMPUS_PHOTO_B64)
    st.image(io.BytesIO(campus_bytes), use_container_width=True)

with header_col2:
    st.markdown(
        """
        <div class="page-title">
            <h1>Keshav Mahavidyalaya PG Finder</h1>
            <p>Browse PGs near Pitampura and get a fair rent estimate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

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
tab_estimator, tab_browse, tab_compare, tab_map, tab_insights, tab_fav, tab_faq = st.tabs(
    ["Rent Estimator", "Browse PGs", "Compare", "Map", "Model Insights", "Favourites", "FAQ"]
)

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
        margin = max(model_metrics["mae"], 400)
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
            "The estimate comes from a Random Forest model trained on sample PG pricing patterns "
            "near the college, factoring in distance, sharing type, area and amenities. See the "
            "'Model Insights' tab for how the model performs and what drives its predictions. "
            "This is illustrative — always confirm the final price with the PG owner."
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
# TAB — MODEL INSIGHTS (the ML chart, in its own section)
# ------------------------------------------------------------
with tab_insights:
    st.subheader("How the rent model works")
    st.write(
        "The Rent Estimator is powered by a Random Forest model trained on 260 simulated PG "
        "profiles. This tab shows how well it performs and what it weighs most heavily — kept "
        "separate from the estimator itself so that tab stays focused on just getting a number."
    )

    ic1, ic2 = st.columns(2)
    with ic1:
        st.metric("R² on held-out data", f"{model_metrics['r2']:.2f}")
    with ic2:
        st.metric("Average error", f"₹{int(model_metrics['mae']):,}")

    st.markdown("**What drives the rent estimate**")
    st.bar_chart(feature_importances)
    st.caption(
        "Higher bars mean the model relies on that factor more when predicting rent. "
        "'Distance' and 'Sharing' (single/double/triple) tend to dominate, with amenities and "
        "area adding smaller adjustments."
    )

    st.markdown("**Average rent by area (from the sample listings)**")
    st.bar_chart(pg_database.groupby("Area")["Monthly Rent (₹)"].mean())

    st.markdown("**Rent vs. distance from college (sample listings)**")
    st.scatter_chart(pg_database, x="Distance (km)", y="Monthly Rent (₹)", color="Area")

    with st.expander("Why train on simulated data instead of real listings?"):
        st.write(
            "There isn't a real, verified dataset of PG rents around the college behind this app "
            "yet — the 15 listings you see in 'Browse PGs' are illustrative, not scraped or sourced "
            "from Google or any real database. The model is trained on synthetic data built from a "
            "reasonable pricing formula so the estimator has something to learn from. If real rent "
            "data becomes available, the model can be retrained on that instead for a genuinely "
            "accurate estimate."
        )

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
        ("Where does the rent model's training data come from?",
         "It's simulated, not scraped from any real source — see the 'Model Insights' tab for details."),
        ("Is the campus photo real?", "Yes — it's the photo you provided of Keshav Mahavidyalaya, embedded "
         "directly in this file so it displays without needing a separate image file."),
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