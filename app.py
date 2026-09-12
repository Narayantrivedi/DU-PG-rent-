"""
Keshav Mahavidyalaya PG Smart Finder & Rent Predictor
------------------------------------------------------
A Streamlit app to help students find nearby PG (paying-guest) accommodation
around Pitampura / Keshav Mahavidyalaya and estimate a fair monthly rent
using a small Machine Learning model.

Run with:  streamlit run pg_finder_app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
# GLOBAL STYLE
# ============================================================
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f9 100%);
    }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #4b3f72 0%, #7c5cbf 45%, #b98dd6 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(75, 63, 114, 0.25);
    }
    .hero h1 {
        margin-bottom: 0.2rem;
        font-size: 1.8rem;
    }
    .hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        text-align: center;
    }

    .pg-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #7c5cbf;
    }
    .pg-card h4 {
        margin: 0 0 0.3rem 0;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 6px;
        background: #f0ecfa;
        color: #4b3f72;
        font-weight: 600;
    }
    .rent-tag {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2f6f4f;
    }
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
        <p>Find verified-style PG listings around Pitampura and get an ML-estimated fair
        monthly rent — built for Keshav Mahavidyalaya students.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MOCK ML TRAINING DATA & MODEL
# ============================================================
# A slightly richer synthetic dataset: distance, sharing type, AC, food included,
# and a one-hot flag for being right in the Pitampura hub (closer to metro/market).
np.random.seed(7)

base_train = pd.DataFrame(
    {
        "Distance": [0.5, 1.2, 2.0, 0.8, 1.5, 2.5, 0.3, 1.0, 0.4, 1.8, 2.2, 0.9, 1.1, 2.8, 0.6, 1.6],
        "Sharing": [1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 3, 2, 2, 3, 1, 2],
        "AC": [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
        "Food": [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "Rent": [12000, 7500, 5500, 13500, 8500, 6000, 14000, 9000,
                 11500, 5200, 6200, 9200, 7800, 4900, 12800, 8800],
    }
)

X = base_train[["Distance", "Sharing", "AC", "Food"]]
y = base_train["Rent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
model = LinearRegression()
model.fit(X_train, y_train)
model_r2 = r2_score(y_test, model.predict(X_test)) if len(X_test) > 1 else None

# Refit on all data for the actual prediction used in the app
final_model = LinearRegression()
final_model.fit(X, y)

# ============================================================
# MOCK PG DATABASE (with extra amenities, ratings & rough coordinates)
# ============================================================
pg_database = pd.DataFrame(
    {
        "PG Name": [
            "Shri Balaji PG",
            "Green View Residency",
            "Pitampura Student Home",
            "Royal Girls Accommodation",
            "Aggarwal PG",
            "Metro Nest PG",
            "Comfort Stay Boys PG",
        ],
        "Area": ["Pitampura", "Rani Bagh", "Kohat Enclave", "Pitampura", "Rani Bagh", "Pitampura", "Kohat Enclave"],
        "Distance (km)": [0.6, 1.4, 2.1, 0.9, 1.6, 0.4, 2.4],
        "Sharing": ["Single / Double", "Double / Triple", "Triple", "Single", "Double", "Single / Double", "Triple"],
        "Gender": ["Boys Only", "Co-ed", "Boys Only", "Girls Only", "Girls Only", "Co-ed", "Boys Only"],
        "Food Included": ["Yes", "Yes", "No", "Yes", "Yes", "No", "No"],
        "AC": ["Yes", "No", "No", "Yes", "Yes", "Yes", "No"],
        "WiFi": ["Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"],
        "Laundry": ["Yes", "No", "No", "Yes", "Yes", "Yes", "No"],
        "Rating": [4.3, 3.9, 3.5, 4.6, 4.0, 4.1, 3.7],
        "Contact": ["+91 98xxxx1234", "+91 98xxxx5678", "+91 98xxxx4321",
                    "+91 98xxxx8765", "+91 98xxxx1122", "+91 98xxxx3344", "+91 98xxxx5566"],
        "Monthly Rent (₹)": [11000, 7800, 5500, 12500, 8200, 12000, 5900],
        "Lat": [28.6985, 28.6889, 28.6928, 28.6992, 28.6875, 28.6970, 28.6940],
        "Lon": [77.1320, 77.1288, 77.1372, 77.1305, 77.1265, 77.1300, 77.1385],
    }
)

# ============================================================
# SIDEBAR — FILTERS
# ============================================================
st.sidebar.header("🔍 Filter Nearby PGs")

search_term = st.sidebar.text_input("Search by PG name")

selected_hub = st.sidebar.selectbox("Preferred Area", ["All Hubs"] + sorted(pg_database["Area"].unique()))
gender_policy = st.sidebar.selectbox("Gender Policy", ["All", "Boys Only", "Girls Only", "Co-ed"])

sharing_filter = st.sidebar.multiselect(
    "Sharing Type",
    options=["Single", "Double", "Triple"],
    default=[],
    help="Leave empty to show all sharing types",
)

food_incl = st.sidebar.checkbox("Mess / Food Included only", value=False)
wifi_only = st.sidebar.checkbox("WiFi only", value=False)
ac_only = st.sidebar.checkbox("AC only", value=False)

min_rent, max_rent = int(pg_database["Monthly Rent (₹)"].min()), int(pg_database["Monthly Rent (₹)"].max())
budget_range = st.sidebar.slider(
    "Budget Range (₹ / month)", min_rent, max_rent, (min_rent, max_rent), step=100
)

sort_by = st.sidebar.selectbox(
    "Sort listings by", ["Distance (nearest first)", "Rent (lowest first)", "Rating (highest first)"]
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
    filtered_df = filtered_df[
        filtered_df["Sharing"].apply(lambda s: any(opt in s for opt in sharing_filter))
    ]

if food_incl:
    filtered_df = filtered_df[filtered_df["Food Included"] == "Yes"]

if wifi_only:
    filtered_df = filtered_df[filtered_df["WiFi"] == "Yes"]

if ac_only:
    filtered_df = filtered_df[filtered_df["AC"] == "Yes"]

filtered_df = filtered_df[
    (filtered_df["Monthly Rent (₹)"] >= budget_range[0])
    & (filtered_df["Monthly Rent (₹)"] <= budget_range[1])
]

if search_term:
    filtered_df = filtered_df[filtered_df["PG Name"].str.contains(search_term, case=False)]

sort_map = {
    "Distance (nearest first)": ("Distance (km)", True),
    "Rent (lowest first)": ("Monthly Rent (₹)", True),
    "Rating (highest first)": ("Rating", False),
}
sort_col, ascending = sort_map[sort_by]
filtered_df = filtered_df.sort_values(sort_col, ascending=ascending)

# ============================================================
# SESSION STATE — FAVOURITES
# ============================================================
if "favourites" not in st.session_state:
    st.session_state.favourites = set()

# ============================================================
# TOP METRICS
# ============================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-card"><h3>{len(filtered_df)}</h3><p>Matching PGs</p></div>',
        unsafe_allow_html=True,
    )
with m2:
    avg_rent = int(filtered_df["Monthly Rent (₹)"].mean()) if not filtered_df.empty else 0
    st.markdown(
        f'<div class="metric-card"><h3>₹{avg_rent:,}</h3><p>Avg. Rent</p></div>',
        unsafe_allow_html=True,
    )
with m3:
    avg_dist = round(filtered_df["Distance (km)"].mean(), 1) if not filtered_df.empty else 0
    st.markdown(
        f'<div class="metric-card"><h3>{avg_dist} km</h3><p>Avg. Distance</p></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><h3>{len(st.session_state.favourites)}</h3><p>⭐ Favourites Saved</p></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["💡 AI Rent Estimator", "🏠 Browse PGs", "🗺️ Map View", "⭐ My Favourites"]
)

# ------------------------------------------------------------
# TAB 1 — AI RENT ESTIMATOR
# ------------------------------------------------------------
with tab1:
    st.subheader("Machine Learning Rent Predictor")
    st.write("Estimate a fair monthly rent based on distance, sharing type, AC and food, "
             "trained on listings around Keshav Mahavidyalaya.")

    col1, col2 = st.columns(2)
    with col1:
        dist_input = st.slider("Distance from Keshav Mahavidyalaya (km)", 0.1, 4.0, 1.0, 0.1)
        sharing_input = st.selectbox(
            "Sharing Type",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Single (1)", 2: "Double (2)", 3: "Triple (3)"}[x],
        )
    with col2:
        ac_input = st.radio("AC Included?", options=["Yes", "No"], horizontal=True)
        food_input = st.radio("Food / Mess Included?", options=["Yes", "No"], horizontal=True)

    ac_val = 1 if ac_input == "Yes" else 0
    food_val = 1 if food_input == "Yes" else 0

    if st.button("Predict Fair Rent", type="primary"):
        prediction = final_model.predict([[dist_input, sharing_input, ac_val, food_val]])[0]
        prediction = max(prediction, 0)
        margin = 600  # illustrative uncertainty band

        st.success(f"Estimated Fair Rent: **₹{int(prediction):,} / month**")
        st.caption(f"Likely range: ₹{int(prediction - margin):,} – ₹{int(prediction + margin):,} / month")

        if model_r2 is not None:
            st.caption(f"Model fit on held-out sample data: R² ≈ {model_r2:.2f} (illustrative, small dataset)")

        # Compare against similar real listings
        similar = pg_database[pg_database["Sharing"].str.contains(
            {1: "Single", 2: "Double", 3: "Triple"}[sharing_input]
        )]
        if not similar.empty:
            st.markdown("**How this compares to similar listings nearby:**")
            compare_df = similar[["PG Name", "Monthly Rent (₹)"]].set_index("PG Name")
            compare_df.loc["Your Estimate"] = int(prediction)
            st.bar_chart(compare_df)

    with st.expander("ℹ️ How is this estimate calculated?"):
        st.write(
            "A linear regression model is trained on sample PG pricing patterns near the college. "
            "It weighs distance from campus, sharing type (single/double/triple), AC availability, "
            "and whether food is included. This is illustrative and meant as a starting reference — "
            "always confirm final pricing directly with the PG owner."
        )

# ------------------------------------------------------------
# TAB 2 — BROWSE PGS
# ------------------------------------------------------------
with tab2:
    st.subheader("Available Listings Near College")

    if filtered_df.empty:
        st.warning("No PGs match your current filters. Try widening your budget or removing a filter.")
    else:
        csv = filtered_df.drop(columns=["Lat", "Lon"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", data=csv, file_name="km_pg_listings.csv", mime="text/csv")

        for _, row in filtered_df.iterrows():
            is_fav = row["PG Name"] in st.session_state.favourites
            with st.container():
                st.markdown('<div class="pg-card">', unsafe_allow_html=True)
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"#### {row['PG Name']}  ·  ⭐ {row['Rating']}")
                    st.markdown(
                        f'<span class="badge">{row["Area"]}</span>'
                        f'<span class="badge">{row["Gender"]}</span>'
                        f'<span class="badge">{row["Sharing"]}</span>'
                        f'<span class="badge">{row["Distance (km)"]} km away</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<span class="rent-tag">₹{row["Monthly Rent (₹)"]:,} / month</span>', unsafe_allow_html=True)
                    amenities = []
                    if row["AC"] == "Yes":
                        amenities.append("❄️ AC")
                    if row["Food Included"] == "Yes":
                        amenities.append("🍽️ Food")
                    if row["WiFi"] == "Yes":
                        amenities.append("📶 WiFi")
                    if row["Laundry"] == "Yes":
                        amenities.append("🧺 Laundry")
                    st.caption(" · ".join(amenities) if amenities else "Basic amenities")
                with c2:
                    fav_label = "💔 Remove" if is_fav else "❤️ Save"
                    if st.button(fav_label, key=f"fav_{row['PG Name']}"):
                        if is_fav:
                            st.session_state.favourites.discard(row["PG Name"])
                        else:
                            st.session_state.favourites.add(row["PG Name"])
                        st.rerun()
                    with st.expander("Contact"):
                        st.write(row["Contact"])
                st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3 — MAP VIEW
# ------------------------------------------------------------
with tab3:
    st.subheader("Where these PGs are, roughly")
    if filtered_df.empty:
        st.info("No listings to show on the map — adjust your filters.")
    else:
        st.map(filtered_df.rename(columns={"Lat": "lat", "Lon": "lon"})[["lat", "lon"]], size=40)
        st.caption("Pin locations are approximate and for general orientation only.")

# ------------------------------------------------------------
# TAB 4 — FAVOURITES
# ------------------------------------------------------------
with tab4:
    st.subheader("Your Saved PGs")
    fav_df = pg_database[pg_database["PG Name"].isin(st.session_state.favourites)]
    if fav_df.empty:
        st.info("You haven't saved any PGs yet. Go to 'Browse PGs' and tap ❤️ Save on a listing.")
    else:
        st.dataframe(
            fav_df.drop(columns=["Lat", "Lon"]),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Built for Keshav Mahavidyalaya students · Listings and rent estimates in this demo are illustrative, "
    "not verified real-time data. Always confirm details directly with PG owners before paying any deposit."
)