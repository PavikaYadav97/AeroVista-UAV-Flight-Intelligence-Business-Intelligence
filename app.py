import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# ============================================================
# AEROVISTA - UAV FLIGHT INTELLIGENCE
# Developed by Pavika Yadav & Amanul Haque
# Bamboo Copter
# ============================================================

st.set_page_config(
    page_title="AeroVista",
    page_icon="🚁",
    layout="wide"
)

# ------------------------------------------------------------
# BAMBOO COPTER THEME
# ------------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b0b0b;
    color: #f2f2f2;
}

.main {
    padding-top: 25px;
}

h1 {
    color: #f5a623 !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

h2, h3 {
    color: #f2f2f2 !important;
}

p, label {
    color: #d5d5d5 !important;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.bc-header {
    border-bottom: 2px solid #f5a623;
    padding-bottom: 15px;
    margin-bottom: 30px;
}

.bc-title {
    font-size: 38px;
    font-weight: 700;
    color: #f5a623;
}

.bc-subtitle {
    font-size: 16px;
    color: #bbbbbb;
    margin-top: 4px;
}

.section-title {
    font-size: 23px;
    font-weight: 600;
    color: #ffffff;
    border-left: 4px solid #f5a623;
    padding-left: 10px;
    margin-top: 30px;
    margin-bottom: 15px;
}

.metric-box {
    background: #151515;
    border: 1px solid #2c2c2c;
    border-radius: 8px;
    padding: 18px;
    min-height: 105px;
}

.metric-label {
    color: #999999;
    font-size: 13px;
}

.metric-value {
    color: #f5a623;
    font-size: 28px;
    font-weight: 600;
    margin-top: 8px;
}

.result-box {
    background: #151515;
    border: 1px solid #333333;
    border-left: 5px solid #f5a623;
    border-radius: 7px;
    padding: 22px;
    margin-top: 15px;
}

.result-normal {
    color: #65c18c;
    font-size: 25px;
    font-weight: 600;
}

.result-warning {
    color: #f5a623;
    font-size: 25px;
    font-weight: 600;
}

.info-box {
    background: #151515;
    border-radius: 7px;
    border: 1px solid #292929;
    padding: 20px;
    line-height: 1.7;
}

.bi-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #242424;
}

.bi-row:last-child {
    border-bottom: none;
}

.bi-label {
    color: #cfcfcf;
    font-size: 15px;
}

.bi-value {
    font-size: 15px;
    font-weight: 600;
}

.bi-ok {
    color: #65c18c;
}

.bi-flag {
    color: #f5a623;
}

.about-box {
    background: #151515;
    border-radius: 7px;
    border: 1px solid #292929;
    padding: 20px 25px;
    line-height: 1.8;
}

.footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #292929;
    text-align: center;
    color: #777777;
    font-size: 13px;
}

.stButton > button {
    background-color: #f5a623;
    color: #111111;
    border: none;
    border-radius: 5px;
    font-weight: 700;
    width: 100%;
    height: 45px;
}

.stButton > button:hover {
    background-color: #ffb83d;
    color: #111111;
}

[data-testid="stMetricValue"] {
    color: #f5a623;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# HTML RENDER HELPER
# ------------------------------------------------------------
# Streamlit's markdown parser treats indented text that follows
# a blank line as a code block. Since every HTML snippet below is
# written inside indented Python code (and some contain blank
# lines for readability), that combination made the browser show
# raw "<p>...</p>" tags instead of rendering them - the bug seen
# in the "Potentially Unusual" box. This helper strips leading
# whitespace from every line before handing it to st.markdown, so
# indentation can never be misread as a code block again.

def render_html(html):
    lines = [line.strip() for line in html.strip().split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ------------------------------------------------------------
# FIND DATASET
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

possible_files = [
    BASE_DIR / "data" / "processed" / "aerovista_analysis_dataset.csv",
    BASE_DIR / "data" / "processed" / "aerovista_master_dataset.csv",
    BASE_DIR / "data" / "raw" / "source2_uav_trajectories.csv"
]

DATA_FILE = None

for file in possible_files:
    if file.exists():
        DATA_FILE = file
        break


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if DATA_FILE is None:
    st.error(
        "AeroVista dataset was not found. "
        "Please check the data/processed folder."
    )
    st.stop()

df = load_data(DATA_FILE)


# ------------------------------------------------------------
# FIND USEFUL DATA COLUMNS
# ------------------------------------------------------------

def find_column(names):
    for name in names:
        if name in df.columns:
            return name
    return None


speed_col = find_column([
    "speed",
    "overall_speed",
    "horizontal_speed"
])

altitude_col = find_column([
    "z",
    "altitude",
    "aircraftAltitude"
])

distance_col = find_column([
    "distance_3d",
    "horizontal_distance"
])

acceleration_col = find_column([
    "acceleration",
    "absolute_acceleration"
])

cluster_col = find_column([
    "flight_cluster",
    "cluster"
])

anomaly_col = find_column([
    "anomaly_label",
    "anomaly"
])


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

render_html("""
<div class="bc-header">
<div class="bc-title">AeroVista</div>
<div class="bc-subtitle">
UAV Flight Intelligence & Business Intelligence
</div>
</div>
""")

st.write(
    "Enter a few basic details about a drone flight. "
    "AeroVista compares the flight with movement patterns found "
    "in the reference UAV dataset."
)


# ------------------------------------------------------------
# REFERENCE DATA SUMMARY
# ------------------------------------------------------------

total_records = len(df)

c1, c2, c3 = st.columns(3)

with c1:
    render_html(f"""
    <div class="metric-box">
        <div class="metric-label">Reference UAV Records</div>
        <div class="metric-value">{total_records:,}</div>
    </div>
    """)

with c2:
    clusters = df[cluster_col].nunique() if cluster_col else 3

    render_html(f"""
    <div class="metric-box">
        <div class="metric-label">Behaviour Groups</div>
        <div class="metric-value">{clusters}</div>
    </div>
    """)

with c3:
    if anomaly_col:
        anomaly_count = (
            df[anomaly_col]
            .astype(str)
            .str.lower()
            .eq("anomaly")
            .sum()
        )
    else:
        anomaly_count = 0

    render_html(f"""
    <div class="metric-box">
        <div class="metric-label">Reference Anomalies</div>
        <div class="metric-value">{anomaly_count:,}</div>
    </div>
    """)


# ------------------------------------------------------------
# ABOUT THIS PROJECT (bullet-point brief for viva / demo)
# ------------------------------------------------------------

with st.expander("ℹ️  About This Project"):

    st.markdown("**Project Overview**")
    st.markdown(f"""
- **Project:** AeroVista — a UAV Flight Intelligence & Business Intelligence system
- **Reference dataset:** {total_records:,} UAV trajectory records (position, speed, altitude and acceleration measurements)
- **Purpose:** compare a user's flight details with movement patterns learned from the reference dataset and return a behaviour group, a pattern assessment, and a comparison summary
""")

    st.markdown("**Behaviour Groups (K-Means Clustering, K = 3)**")

    if cluster_col and cluster_col in df.columns:
        cluster_counts = df[cluster_col].value_counts().sort_values(ascending=False)

        group_labels = [
            "Normal / Common Flight Behaviour",
            "Alternative Flight Behaviour",
            "Less Common Flight Behaviour"
        ]
        group_descriptions = [
            "Represents the largest group in the dataset. These records reflect the most frequently occurring movement pattern in the reference UAV data. Flights with characteristics close to this group's centroid are assigned here.",
            "Represents another distinct movement pattern found in the UAV data, differing from the dominant cluster above. A new flight is assigned here when its calculated movement characteristics are closer to this cluster's centroid.",
            "The smallest of the three groups, representing a relatively less frequently observed movement pattern in the reference dataset. A flight is assigned here when its characteristics are closest to this cluster's centroid."
        ]

        for rank, (cluster_id, count) in enumerate(cluster_counts.items()):
            st.markdown(f"**Cluster {cluster_id} — {group_labels[rank]}**")
            st.markdown(f"""
- Contains **{count:,} records**
- {group_descriptions[rank]}
""")
    else:
        st.markdown("- Cluster sizes will appear once the dataset includes a cluster column.")

    st.markdown(
        "- Note: these clusters are mathematical groupings based on similarity in movement — "
        "the cluster number itself does not mean \"normal\" or \"abnormal\" on its own. "
        "That judgement comes separately from anomaly detection."
    )

    st.markdown("**Project Flow**")
    st.markdown("""
- Collect and clean raw UAV trajectory data (position, speed, altitude, acceleration)
- Engineer movement features such as speed, 3D distance, vertical speed and acceleration
- Apply K-Means clustering (K = 3) to group flights by movement similarity
- Apply anomaly detection to flag observations that sit well outside the learned patterns
- Build an interactive Streamlit dashboard so a user can enter basic flight details
- Compare the entered flight against the reference dataset and the trained clusters
- Return a behaviour group, a within/outside-pattern assessment, and a visual comparison
""")

    st.markdown("**Technology Stack**")
    st.markdown("""
- **Python** — core data processing and analysis
- **Pandas & NumPy** — data cleaning and feature engineering
- **Scikit-learn** — K-Means clustering and anomaly detection
- **Streamlit** — interactive web dashboard
""")

    st.markdown("**Developed by**")
    st.markdown("Pavika Yadav & Amanul Haque — Bamboo Copter")


# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

render_html('<div class="section-title">Enter Flight Details</div>')

st.write(
    "Only basic flight information is required. "
    "The remaining analytical values are calculated automatically."
)

col1, col2, col3 = st.columns(3)

with col1:
    distance_input = st.number_input(
        "Distance travelled (metres)",
        min_value=1.0,
        max_value=10000.0,
        value=500.0,
        step=10.0
    )

with col2:
    duration_input = st.number_input(
        "Flight duration (seconds)",
        min_value=1.0,
        max_value=36000.0,
        value=120.0,
        step=5.0
    )

with col3:
    altitude_input = st.number_input(
        "Maximum altitude (metres)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=5.0
    )

st.write("")

analyse = st.button("ANALYSE FLIGHT")


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

if analyse:

    # Basic flight calculations
    flight_speed = distance_input / duration_input

    vertical_speed = altitude_input / duration_input

    # Simple 3D movement estimate
    distance_3d = np.sqrt(
        distance_input ** 2 +
        altitude_input ** 2
    )

    overall_speed = distance_3d / duration_input

    # --------------------------------------------------------
    # REFERENCE VALUES
    # --------------------------------------------------------

    reference_speed = (
        pd.to_numeric(df[speed_col], errors="coerce").dropna()
        if speed_col else pd.Series([flight_speed])
    )

    reference_altitude = (
        pd.to_numeric(df[altitude_col], errors="coerce").dropna()
        if altitude_col else pd.Series([altitude_input])
    )

    reference_distance = (
        pd.to_numeric(df[distance_col], errors="coerce").dropna()
        if distance_col else pd.Series([distance_3d])
    )

    # Remove impossible values
    reference_speed = reference_speed[
        np.isfinite(reference_speed)
    ]

    reference_altitude = reference_altitude[
        np.isfinite(reference_altitude)
    ]

    reference_distance = reference_distance[
        np.isfinite(reference_distance)
    ]

    avg_speed = reference_speed.mean()
    avg_altitude = reference_altitude.mean()
    avg_distance = reference_distance.mean()

    # --------------------------------------------------------
    # DETERMINE WHETHER INPUT IS WITHIN REFERENCE RANGE
    # --------------------------------------------------------

    speed_low = reference_speed.quantile(0.05)
    speed_high = reference_speed.quantile(0.95)

    altitude_low = reference_altitude.quantile(0.05)
    altitude_high = reference_altitude.quantile(0.95)

    distance_low = reference_distance.quantile(0.05)
    distance_high = reference_distance.quantile(0.95)

    speed_outside = (
        flight_speed < speed_low or
        flight_speed > speed_high
    )

    altitude_outside = (
        altitude_input < altitude_low or
        altitude_input > altitude_high
    )

    distance_outside = (
        distance_3d < distance_low or
        distance_3d > distance_high
    )

    unusual_count = sum([
        speed_outside,
        altitude_outside,
        distance_outside
    ])

    # --------------------------------------------------------
    # FLIGHT ASSESSMENT
    # --------------------------------------------------------

    if unusual_count >= 2:
        assessment = "Potentially Unusual"
        assessment_class = "result-warning"
    else:
        assessment = "Within Reference Pattern"
        assessment_class = "result-normal"

    # --------------------------------------------------------
    # FIND CLOSEST BEHAVIOUR GROUP
    # --------------------------------------------------------

    if cluster_col and speed_col and altitude_col:

        model_data = df[
            [speed_col, altitude_col]
        ].copy()

        model_data[speed_col] = pd.to_numeric(
            model_data[speed_col],
            errors="coerce"
        )

        model_data[altitude_col] = pd.to_numeric(
            model_data[altitude_col],
            errors="coerce"
        )

        model_data = model_data.dropna()

        # Limit training size for faster dashboard response
        if len(model_data) > 30000:
            model_data = model_data.sample(
                30000,
                random_state=42
            )

        scaler = StandardScaler()

        X = scaler.fit_transform(
            model_data[[speed_col, altitude_col]]
        )

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)

        user_scaled = scaler.transform(
            [[flight_speed, altitude_input]]
        )

        predicted_cluster = int(
            kmeans.predict(user_scaled)[0]
        )

        # Give clusters a readable name
        cluster_name = f"Behaviour Group {predicted_cluster + 1}"

    else:
        predicted_cluster = 0
        cluster_name = "Behaviour Group 1"

    # --------------------------------------------------------
    # BUSINESS INSIGHT
    # --------------------------------------------------------

    if assessment == "Potentially Unusual":

        insight = (
            "The entered flight differs from the normal range "
            "observed in the reference UAV data. "
            "The flight may require additional review."
        )

    else:

        insight = (
            "The entered flight falls within the movement range "
            "observed in the reference UAV data. "
            "Its overall behaviour is similar to the reference flights."
        )

    def bi_flag(is_outside):
        return (
            '<span class="bi-flag">Outside typical range</span>'
            if is_outside else
            '<span class="bi-ok">Within typical range</span>'
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    render_html('<div class="section-title">Flight Analysis</div>')

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        render_html(f"""
        <div class="metric-box">
            <div class="metric-label">Estimated Speed</div>
            <div class="metric-value">{flight_speed:.2f} m/s</div>
        </div>
        """)

    with r2:
        render_html(f"""
        <div class="metric-box">
            <div class="metric-label">Behaviour Group</div>
            <div class="metric-value">{predicted_cluster + 1}</div>
        </div>
        """)

    with r3:
        render_html(f"""
        <div class="metric-box">
            <div class="metric-label">Altitude</div>
            <div class="metric-value">{altitude_input:.1f} m</div>
        </div>
        """)

    with r4:
        render_html(f"""
        <div class="metric-box">
            <div class="metric-label">Reference Records</div>
            <div class="metric-value">{total_records:,}</div>
        </div>
        """)

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    render_html(f"""
    <div class="result-box">
        <div class="{assessment_class}">{assessment}</div>
        <p style="margin-top:10px;">
        AeroVista placed this flight in <b>{cluster_name}</b>
        based on its movement characteristics.
        </p>
    </div>
    """)

    # --------------------------------------------------------
    # BUSINESS INTELLIGENCE
    # --------------------------------------------------------

    render_html('<div class="section-title">Business Intelligence</div>')

    render_html(f"""
    <div class="info-box">
        <b>Flight assessment</b><br><br>
        {insight}
        <br><br>
        <b>Reference comparison</b>
        <div class="bi-row">
            <span class="bi-label">Speed — {flight_speed:.2f} m/s vs reference avg {avg_speed:.2f} m/s</span>
            <span class="bi-value">{bi_flag(speed_outside)}</span>
        </div>
        <div class="bi-row">
            <span class="bi-label">Altitude — {altitude_input:.1f} m vs reference avg {avg_altitude:.2f} m</span>
            <span class="bi-value">{bi_flag(altitude_outside)}</span>
        </div>
        <div class="bi-row">
            <span class="bi-label">3D Distance — {distance_3d:.2f} m vs reference avg {avg_distance:.2f} m</span>
            <span class="bi-value">{bi_flag(distance_outside)}</span>
        </div>
    </div>
    """)

    # --------------------------------------------------------
    # COMPARISON CHART
    # --------------------------------------------------------

    render_html('<div class="section-title">Your Flight vs Reference Data</div>')

    comparison = pd.DataFrame({
        "Your Flight": [
            flight_speed,
            altitude_input,
            distance_3d
        ],
        "Reference Average": [
            avg_speed,
            avg_altitude,
            avg_distance
        ]
    }, index=[
        "Speed",
        "Altitude",
        "3D Distance"
    ])

    st.bar_chart(comparison)

    # --------------------------------------------------------
    # SIMPLE SUMMARY
    # --------------------------------------------------------

    render_html('<div class="section-title">Flight Summary</div>')

    summary = pd.DataFrame({
        "Parameter": [
            "Distance travelled",
            "Flight duration",
            "Maximum altitude",
            "Estimated speed",
            "Estimated 3D distance",
            "Behaviour group",
            "Assessment"
        ],

        "Value": [
            f"{distance_input:.1f} m",
            f"{duration_input:.1f} sec",
            f"{altitude_input:.1f} m",
            f"{flight_speed:.2f} m/s",
            f"{distance_3d:.2f} m",
            f"Group {predicted_cluster + 1}",
            assessment
        ]
    })

    st.table(summary)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

render_html("""
<div class="footer">
AeroVista — UAV Flight Intelligence<br>
Data Mining & Business Intelligence Project<br><br>
Developed by <b>Pavika Yadav & Amanul Haque</b><br>
Bamboo Copter
</div>
""")