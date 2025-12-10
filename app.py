import streamlit as st
import pandas as pd
import pickle
import numpy as np
from io import StringIO
import os

st.set_page_config(
    page_title="NBA MVP Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Overall page */
        .main {
            background: #f5f5f5;
            color: #111827;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #ffffff;
            color: #111827;
            border-right: 1px solid #e5e7eb;
        }

        /* Headline banner */
        .mvp-header-bar {
            background: linear-gradient(90deg, #ffffff 0%, #f3f4f6 50%, #e5e7eb 100%);
            padding: 0.9rem 1.2rem;
            border-radius: 0.75rem;
            border: 1px solid #d1d5db;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .mvp-header-title {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #111827;
        }

        .mvp-header-subtitle {
            font-size: 0.9rem;
            color: #4b5563;
            max-width: 520px;
        }

        /* Section labels */
        .mvp-section-label {
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.75rem;
            color: #6b7280;
        }

        /* Headings */
        h1, h2, h3 {
            color: #111827 !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #ffffff;
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
        }

        /* Dataframe */
        .stDataFrame {
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #dc2626, #f97316);
            color: #ffffff;
            border-radius: 999px;
            padding: 0.55rem 1.7rem;
            border: none;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .stButton>button:hover {
            filter: brightness(1.05);
        }

        /* Info / success boxes */
        .stAlert {
            border-radius: 0.75rem;
        }

        /* Section dividers */
        hr {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 1.5rem 0 1rem 0;
        }

        /* Small text */
        .subtle {
            color: #6b7280;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "mvp_model.pkl")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model_pkg = load_model()
    model = model_pkg["model"]
    feature_cols = model_pkg["feature_cols"]
except Exception as e:
    st.error(
        "Unable to load the MVP model. "
        "Please ensure `mvp_model.pkl` is in the same directory as this app."
    )
    st.stop()

with st.sidebar:
    st.markdown("## 🏀 NBA MVP Predictor")
    st.markdown(
        "Use advanced box score and team metrics to predict the **NBA MVP**. "
        "Each player in the dataset will be given a MVP score."
    )

    st.markdown("---")
    st.markdown("### 📂 How to Use")
    st.markdown(
        """
        1. Export a CSV of player stats.
        2. Make sure it contains the required columns:
           - `Player`, `Team`, `Age`, `Pos`, `G`, `MP`
           - `PTS`, `TRB`, `AST`, `STL`, `BLK`
           - `FGA`, `FTA`, `TOV`
           - `FG%`, `3P%`, `eFG%`, `FT%`
           - `W`, `L`, `PS/G`, `PA/G`
        3. Upload the CSV using the file uploader.
        4. Click **Predict MVP** to generate scores and rankings.
        """
    )

    st.markdown("---")
    st.markdown("### ℹ️ About the Model")
    st.markdown(
        f"""
        - **Model:** `{model_pkg.get('model_name', 'Unknown')}`
        - **Training Window:** `{model_pkg.get('training_years', 'N/A')}`
        """
    )
    if "performance" in model_pkg:
        perf = model_pkg["performance"]
        st.markdown("**Validation Performance**")
        st.metric("RMSE", f"{perf['RMSE']:.2f}")
        st.metric("MAE", f"{perf['MAE']:.2f}")
        st.metric("R²", f"{perf['R2']:.4f}")

    st.markdown("---")
    st.markdown(
        "<p class='subtle'>Built with Streamlit • Trained on 1980–2022 NBA seasons</p>",
        unsafe_allow_html=True,
    )

st.title("🏆 NBA MVP Predictor Dashboard")

st.markdown(
    """
    A dashboard inspired by ESPN's player pages.
    Upload a season-level stats file and get an analytically driven
    ranking of **Most Valuable Player** candidates.
    """
)

st.markdown("---")

# -----------------------------------------------------------------------------
# File Upload Section
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📥 Upload Player Statistics CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV containing season-level player statistics",
        type="csv",
        label_visibility="collapsed",
    )

with col_right:
    st.markdown("#### Required Columns")
    st.markdown(
        """
        Box‑score & context columns:
        - `Player`, `Team`, `Age`, `Pos`, `G`, `MP`
        - `PTS`, `TRB`, `AST`, `STL`, `BLK`
        - `FGA`, `FTA`, `TOV`
        - `FG%`, `3P%`, `eFG%`, `FT%`
        - `W`, `L`, `PS/G`, `PA/G`
        """
    )

if uploaded_file is None:
    st.info(
        "👆 Upload a CSV file with NBA player statistics to get started.\n\n"
        "You will see a preview of your data and a ranked list of MVP candidates."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file as CSV. Error: {e}")
    st.stop()

st.subheader("📊 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

missing_core_cols = [
    col
    for col in [
        "Player",
        "Team",
        "Age",
        "Pos",
        "G",
        "MP",
        "PTS",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "FGA",
        "FTA",
        "TOV",
        "FG%",
        "3P%",
        "eFG%",
        "FT%",
        "W",
        "L",
        "PS/G",
        "PA/G",
    ]
    if col not in df.columns
]

if missing_core_cols:
    st.warning(
        "Some important columns are missing from your data:\n\n"
        + ", ".join(f"`{c}`" for c in missing_core_cols)
        + "\n\nThe model will still run but predictions may be less reliable."
    )

st.markdown("---")

center_col = st.columns([1, 1, 1])[1]
with center_col:
    run_prediction = st.button("🔮 Predict MVP", type="primary", use_container_width=True)

if not run_prediction:
    st.stop()

with st.spinner("Computing advanced metrics and generating MVP scores..."):
    df = df.copy()
    df = df.fillna(0)

    # Per-game stats (data already contains per-game values)
    df["PPG"] = df.get("PTS", 0)
    df["RPG"] = df.get("TRB", 0)
    df["APG"] = df.get("AST", 0)
    df["SPG"] = df.get("STL", 0)
    df["BPG"] = df.get("BLK", 0)
    df["MPG"] = df.get("MP", 0)
    df["TPG"] = df.get("TOV", 0)

    # Shooting & usage metrics
    fga = df.get("FGA", 0)
    fta = df.get("FTA", 0)
    mp = df.get("MP", 0).replace(0, 1)

    df["TS%"] = df.get("PTS", 0) / (2 * (fga + 0.44 * fta)).replace(0, 1)
    df["Usage"] = (fga + 0.44 * fta + df.get("TOV", 0)) / mp * 100

    # Team win rate
    w = df.get("W", 0)
    l = df.get("L", 0)
    df["Win_Rate"] = w / (w + l).replace(0, 1)

    # Position dummies if available
    if "Pos" in df.columns:
        df["Primary_Pos"] = df["Pos"].astype(str).str.split("-").str[0]
        pos_dummies = pd.get_dummies(df["Primary_Pos"], prefix="Pos")
        df = pd.concat([df, pos_dummies], axis=1)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Final feature matrix
    X = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # Predict MVP scores
    predictions = model.predict(X)
    df["MVP_Score"] = predictions
    df["MVP_Rank"] = df["MVP_Score"].rank(ascending=False, method="first")

    results = df.sort_values("MVP_Score", ascending=False).reset_index(drop=True)

st.success("✅ Predictions complete!")

st.subheader("🏅 Top 5 MVP Candidates")

top_5_results = results.head(5)

if len(top_5_results) > 0:
    # Create a tab for each of the top 5 players
    tab_labels = [name for name in top_5_results["Player"]]
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            player_data = top_5_results.iloc[i]
            
            # Player header
            st.markdown(f"### {player_data['Player']}")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MVP Rank", f"#{int(player_data['MVP_Rank'])}")
            with col2:
                st.metric("Team", str(player_data.get("Team", "N/A")))
            with col3:
                if "W" in player_data and "L" in player_data:
                    st.metric("Team Record", f"{int(player_data['W'])}-{int(player_data['L'])}")
                else:
                    st.metric("Team Record", "N/A")
            with col4:
                st.metric("MVP Score", f"{player_data['MVP_Score']:.1f}", help="Model-generated score indicating MVP likelihood.")

            st.markdown("---")
            
            # Per-game stats
            st.markdown("##### Per-Game Averages")
            pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
            with pcol1:
                st.metric("Points (PPG)", f"{player_data.get('PPG', 0):.1f}")
            with pcol2:
                st.metric("Rebounds (RPG)", f"{player_data.get('RPG', 0):.1f}")
            with pcol3:
                st.metric("Assists (APG)", f"{player_data.get('APG', 0):.1f}")
            with pcol4:
                st.metric("Steals (SPG)", f"{player_data.get('SPG', 0):.1f}")
            with pcol5:
                st.metric("Blocks (BPG)", f"{player_data.get('BPG', 0):.1f}")
            
            st.markdown("")
            
            # Shooting Efficiency
            st.markdown("##### Shooting Efficiency")
            scol1, scol2, scol3, scol4, scol5 = st.columns(5)
            with scol1:
                st.metric("FG%", f"{player_data.get('FG%', 0):.1%}" if player_data.get('FG%', 0) <= 1 else f"{player_data.get('FG%', 0):.1f}%")
            with scol2:
                st.metric("3P%", f"{player_data.get('3P%', 0):.1%}" if player_data.get('3P%', 0) <= 1 else f"{player_data.get('3P%', 0):.1f}%")
            with scol3:
                st.metric("eFG%", f"{player_data.get('eFG%', 0):.1%}" if player_data.get('eFG%', 0) <= 1 else f"{player_data.get('eFG%', 0):.1f}%")
            with scol4:
                st.metric("FT%", f"{player_data.get('FT%', 0):.1%}" if player_data.get('FT%', 0) <= 1 else f"{player_data.get('FT%', 0):.1f}%")
            with scol5:
                st.metric("TS%", f"{player_data.get('TS%', 0):.1%}" if player_data.get('TS%', 0) <= 1 else f"{player_data.get('TS%', 0):.1f}%")
            
            st.markdown("")
            
            # Advanced Metrics
            st.markdown("##### Advanced Metrics")
            acol1, acol2, acol3 = st.columns(3)
            with acol1:
                st.metric("Usage Rate", f"{player_data.get('Usage', 0):.1f}%")
            with acol2:
                st.metric("Team Win Rate", f"{player_data.get('Win_Rate', 0):.1%}" if player_data.get('Win_Rate', 0) <= 1 else f"{player_data.get('Win_Rate', 0):.1f}%")
            with acol3:
                st.metric("Age", f"{int(player_data.get('Age', 0))}" if player_data.get('Age', 0) > 0 else "N/A")

else:
    st.info("Run a prediction to see the top MVP candidates here.")


if len(results) > 0:
    st.markdown("---")
    st.subheader("👑 Predicted MVP")

    mvp = results.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Player", str(mvp.get("Player", "N/A")))
    with col2:
        st.metric("Team", str(mvp.get("Team", "N/A")))
    with col3:
        if "W" in mvp and "L" in mvp:
            st.metric("Team Record", f"{int(mvp['W'])}-{int(mvp['L'])}")
        else:
            st.metric("Team Record", "N/A")
    with col4:
        st.metric("MVP Score", f"{mvp['MVP_Score']:.1f}")

    if all(col in mvp for col in ["PPG", "RPG", "APG"]):
        st.markdown(
            f"**Per-game stats:** {mvp['PPG']:.1f} PPG • {mvp['RPG']:.1f} RPG • {mvp['APG']:.1f} APG"
        )

st.markdown("---")
st.subheader("📥 Download Full Prediction Results")

csv_buf = StringIO()
# Use a clean, export-focused set of columns
export_cols = [
    "MVP_Rank",
    "Player",
    "Team",
    "Age",
    "Pos",
    "G",
    "GS",
    "MP",
    "PPG",
    "RPG",
    "APG",
    "SPG",
    "BPG",
    "TPG",
    "TS%",
    "Usage",
    "W",
    "L",
    "Win_Rate",
    "MVP_Score",
]
export_cols = [c for c in export_cols if c in results.columns]
results[export_cols].to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode("utf-8")

st.download_button(
    label="Download CSV of MVP Rankings",
    data=csv_bytes,
    file_name="mvp_predictions.csv",
    mime="text/csv",
    type="primary",
)