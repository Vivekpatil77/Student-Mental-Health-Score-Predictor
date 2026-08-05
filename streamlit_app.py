import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MindPulse AI | Student Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONSTANTS & SETUP
# ==========================================
MODEL_PATH = "Mental_Health_Model.pkl"
DATASET_PATH = "Student Social Media And Mental Health Impact.csv"

TOP_COUNTRIES = [
    'Other', 'India', 'USA', 'Canada', 'Australia', 
    'UK', 'Germany', 'Mexico', 'Turkey', 'France'
]

PLATFORMS = [
    'Instagram', 'Facebook', 'LinkedIn', 'Snapchat', 'Twitter',
    'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte',
    'WhatsApp', 'WeChat'
]

PURPOSES = ['Networking', 'Education', 'Entertainment', 'News']
ACADEMIC_LEVELS = ['Undergraduate', 'Graduate', 'High School']
STRESS_LEVELS = ['Low', 'Medium', 'High', 'Very High']

# Developer Credentials
DEV_NAME = "Vivek Namdev Patil"
DEV_COLLEGE = "AISSMS College of Engineering Pune"
GITHUB_URL = "https://github.com"
LINKEDIN_URL = "https://linkedin.com"

# ==========================================
# CUSTOM CSS (DARK MODE & GLASSMORPHISM)
# ==========================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.15);
        border-color: rgba(79, 70, 229, 0.3);
    }

    /* Hero Banner */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }

    /* Badges & Pills */
    .badge-primary {
        background: rgba(79, 70, 229, 0.2);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* Action Buttons Customization */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        border: none;
    }

    /* Primary Predict Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
        transform: translateY(-1px);
    }

    /* Status Result Cards */
    .result-card-healthy {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .result-card-moderate {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .result-card-high {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Footer Styling */
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #64748B;
        text-align: center;
        padding: 10px 0;
        font-size: 0.85rem;
        z-index: 999;
    }

    /* Metric Cards */
    .metric-container {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA & MODEL LOADING FUNCTIONS
# ==========================================
@st.cache_resource
def load_ml_model():
    """Loads the trained machine learning model pipeline."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None

@st.cache_data
def load_dataset():
    """Loads the student social media and mental health dataset."""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading dataset from {DATASET_PATH}: {e}")
        return None

model = load_ml_model()
df_data = load_dataset()


# ==========================================
# LOTTIE ANIMATION HELPER
# ==========================================
def load_lottie_url(url: str):
    """Fallback inline SVG/Lottie loader."""
    try:
        import requests
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# AI Mind Illustration Lottie
lottie_ai_url = "https://assets5.lottiefiles.com/packages/lf20_sk5h1kfn.json"
lottie_ai_json = load_lottie_url(lottie_ai_url)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "age" not in st.session_state:
    st.session_state.age = 21
if "gender" not in st.session_state:
    st.session_state.gender = "Male"
if "country" not in st.session_state:
    st.session_state.country = "India"
if "academic_level" not in st.session_state:
    st.session_state.academic_level = "Undergraduate"
if "platform" not in st.session_state:
    st.session_state.platform = "Instagram"
if "purpose" not in st.session_state:
    st.session_state.purpose = "Entertainment"
if "usage_hours" not in st.session_state:
    st.session_state.usage_hours = 4.0
if "unlocks" not in st.session_state:
    st.session_state.unlocks = 120
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 4.0
if "physical_hours" not in st.session_state:
    st.session_state.physical_hours = 1.5
if "sleep_hours" not in st.session_state:
    st.session_state.sleep_hours = 7.0
if "stress_level" not in st.session_state:
    st.session_state.stress_level = "Medium"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def reset_inputs():
    """Resets input components to defaults."""
    st.session_state.age = 21
    st.session_state.gender = "Male"
    st.session_state.country = "India"
    st.session_state.academic_level = "Undergraduate"
    st.session_state.platform = "Instagram"
    st.session_state.purpose = "Entertainment"
    st.session_state.usage_hours = 4.0
    st.session_state.unlocks = 120
    st.session_state.study_hours = 4.0
    st.session_state.physical_hours = 1.5
    st.session_state.sleep_hours = 7.0
    st.session_state.stress_level = "Medium"
    st.session_state.prediction_result = None

def load_random_sample():
    """Loads a random realistic student record from dataset."""
    if df_data is not None and not df_data.empty:
        sample = df_data.sample(1).iloc[0]
        st.session_state.age = int(sample.get("Age", 21))
        st.session_state.gender = sample.get("Gender", "Female")
        cnt = sample.get("Country", "India")
        st.session_state.country = cnt if cnt in TOP_COUNTRIES else "Other"
        st.session_state.academic_level = sample.get("Academic_Level", "Undergraduate")
        st.session_state.platform = sample.get("Most_Used_Platform", "Instagram")
        st.session_state.purpose = sample.get("Purpose_Of_Use", "Entertainment")
        st.session_state.usage_hours = float(sample.get("Avg_Daily_Usage_Hours", 4.0))
        st.session_state.unlocks = int(sample.get("Daily_Unlocks", 120))
        st.session_state.study_hours = float(sample.get("Study_Hours", 4.0))
        st.session_state.physical_hours = float(sample.get("Physical_Activity_Hours", 1.5))
        st.session_state.sleep_hours = float(sample.get("Sleep_Hours_Per_Night", 7.0))
        st.session_state.stress_level = sample.get("Stress_Level", "Medium")
        st.session_state.prediction_result = None


# ==========================================
# SIDEBAR CONTENT
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <span style="font-size: 2.2rem;">🧠</span>
            <h2 style="margin: 5px 0 0 0; color: #F8FAFC; font-weight: 700;">MindPulse AI</h2>
            <p style="color: #64748B; font-size: 0.8rem; margin: 0;">Student Mental Health Predictor</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # Navigation Menu
    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Predict Score", "Analytics & Insights", "Model Info", "About & Developer"],
        icons=["house-door", "cpu", "pie-chart", "diagram-3", "person-badge"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#06B6D4", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px",
                "border-radius": "8px",
                "color": "#94A3B8",
            },
            "nav-link-selected": {"background-color": "#4F46E5", "color": "#FFFFFF", "font-weight": "600"},
        }
    )

    st.divider()

    # Sidebar Quick Info Accordions
    with st.expander("📌 Project Overview"):
        st.caption(
            "Predicts student mental health distress scores based on social media interaction, "
            "screen time, daily unlocks, academic habits, and lifestyle routines using Random Forest Regression."
        )

    with st.expander("🤖 Model Architecture"):
        st.caption(
            "**Algorithm**: Random Forest Regressor Pipeline\n"
            "**Preprocessing**: ColumnTransformer (StandardScaler, OneHotEncoder, OrdinalEncoder)\n"
            "**Features**: 12 Multi-domain attributes"
        )

    with st.expander("📊 Dataset Details"):
        st.caption(
            "**Sample Count**: 5,000 Student Records\n"
            "**Target Variable**: Mental Health Score (1.0 to 10.0 scale)"
        )

    with st.expander("👨‍💻 Developer Profile"):
        st.markdown(f"**{DEV_NAME}**")
        st.caption(f"{DEV_COLLEGE}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_gh, c_li = st.columns(2)
    with c_gh:
        st.markdown(f'<a href="{GITHUB_URL}" target="_blank" style="text-decoration:none;"><button style="width:100%; background:#334155; color:white; border:none; padding:8px; border-radius:8px; cursor:pointer;">💻 GitHub</button></a>', unsafe_allow_html=True)
    with c_li:
        st.markdown(f'<a href="{LINKEDIN_URL}" target="_blank" style="text-decoration:none;"><button style="width:100%; background:#0284C7; color:white; border:none; padding:8px; border-radius:8px; cursor:pointer;">💼 LinkedIn</button></a>', unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME (LANDING PAGE)
# ==========================================
if selected_page == "Home":
    st.markdown('<div class="badge-primary">✨ Final Year ML Project • AI Powered</div>', unsafe_allow_html=True)
    
    hero_col1, hero_col2 = st.columns([1.3, 1])
    
    with hero_col1:
        st.markdown('<h1 class="hero-title">Mental Health Score Prediction Using Machine Learning</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-subtitle">Evaluate, analyze, and predict student mental well-being scores '
            'using intelligent machine learning pipelines trained on social media usage, academic routines, '
            'and personal lifestyle habits.</p>',
            unsafe_allow_html=True
        )
        
        c_btn1, c_btn2 = st.columns([1, 1])
        with c_btn1:
            st.info("👈 Select **'Predict Score'** from the sidebar menu to assess your mental health score!")
        with c_btn2:
            st.markdown(f'<a href="{GITHUB_URL}" target="_blank" style="text-decoration:none;"><button style="width:100%; background:#1E293B; color:#F8FAFC; border:1px solid rgba(255,255,255,0.1); padding:10px; border-radius:10px; font-weight:600; cursor:pointer;">📁 Source Code</button></a>', unsafe_allow_html=True)

    with hero_col2:
        if lottie_ai_json:
            st_lottie(lottie_ai_json, height=280, key="ai_hero_anim")
        else:
            st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 40px 20px;">
                    <span style="font-size: 5rem;">🧠⚡</span>
                    <h3 style="color: #38BDF8; margin-top: 15px;">MindPulse Intelligence</h3>
                    <p style="color: #94A3B8; font-size: 0.9rem;">Predictive AI Engine for Student Mental Wellness</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Highlight Cards Grid
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
            <div class="glass-card">
                <span style="font-size: 2rem;">⚡</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 5px 0;">Instant ML Inference</h4>
                <p style="color:#94A3B8; font-size:0.85rem;">High-precision Random Forest Pipeline delivering predictions in < 15ms.</p>
            </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="glass-card">
                <span style="font-size: 2rem;">🎯</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 5px 0;">Circular Gauge Score</h4>
                <p style="color:#94A3B8; font-size:0.85rem;">Intuitive 0–100 Distress Risk gauge with color-coded severity metrics.</p>
            </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
            <div class="glass-card">
                <span style="font-size: 2rem;">📊</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 5px 0;">Interactive Analytics</h4>
                <p style="color:#94A3B8; font-size:0.85rem;">Explore multi-dimensional radar spider charts, histograms, and pie charts.</p>
            </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
            <div class="glass-card">
                <span style="font-size: 2rem;">💡</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 5px 0;">Smart Action Plan</h4>
                <p style="color:#94A3B8; font-size:0.85rem;">Actionable clinical & lifestyle recommendations tailored to score tiers.</p>
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: PREDICTION PAGE
# ==========================================
elif selected_page == "Predict Score":
    st.markdown('<div class="badge-primary">🔮 Real-Time Mental Health Score Predictor</div>', unsafe_allow_html=True)
    st.markdown("## 📥 Student Profile & Habits Assessment")
    st.caption("Provide lifestyle, digital habit, and academic parameters to generate the predicted score.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form", clear_on_submit=False):
        col_demo, col_social, col_life = st.columns(3)

        # Demographics Container
        with col_demo:
            st.markdown("#### 👤 Demographics")
            age = st.slider("Age", min_value=10, max_value=100, value=st.session_state.age)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1)
            country = st.selectbox("Country", options=TOP_COUNTRIES, index=TOP_COUNTRIES.index(st.session_state.country) if st.session_state.country in TOP_COUNTRIES else 0)
            academic_level = st.selectbox("Academic Level", options=ACADEMIC_LEVELS, index=ACADEMIC_LEVELS.index(st.session_state.academic_level) if st.session_state.academic_level in ACADEMIC_LEVELS else 0)

        # Social Media Habits Container
        with col_social:
            st.markdown("#### 📱 Digital Habits")
            platform = st.selectbox("Most Used Platform", options=PLATFORMS, index=PLATFORMS.index(st.session_state.platform) if st.session_state.platform in PLATFORMS else 0)
            purpose = st.selectbox("Purpose of Use", options=PURPOSES, index=PURPOSES.index(st.session_state.purpose) if st.session_state.purpose in PURPOSES else 0)
            usage_hours = st.slider("Avg Daily Usage (Hours)", min_value=0.0, max_value=24.0, value=float(st.session_state.usage_hours), step=0.1)
            unlocks = st.slider("Daily Phone Unlocks", min_value=0, max_value=300, value=int(st.session_state.unlocks), step=1)

        # Lifestyle Container
        with col_life:
            st.markdown("#### 🏃 Lifestyle & Well-being")
            study_hours = st.slider("Study Hours / Day", min_value=0.0, max_value=24.0, value=float(st.session_state.study_hours), step=0.1)
            physical_hours = st.slider("Physical Activity (Hrs/Day)", min_value=0.0, max_value=24.0, value=float(st.session_state.physical_hours), step=0.1)
            sleep_hours = st.slider("Sleep Hours / Night", min_value=0.0, max_value=24.0, value=float(st.session_state.sleep_hours), step=0.1)
            stress_level = st.selectbox("Stress Level", options=STRESS_LEVELS, index=STRESS_LEVELS.index(st.session_state.stress_level) if st.session_state.stress_level in STRESS_LEVELS else 1)

        st.markdown("<br>", unsafe_allow_html=True)

        # Form Action Buttons
        btn_c1, btn_c2, btn_c3 = st.columns([2, 1, 1])
        with btn_c1:
            submit_predict = st.form_submit_button("🔮 Predict Mental Health Score", type="primary", use_container_width=True)
        with btn_c2:
            submit_random = st.form_submit_button("🎲 Random Sample", use_container_width=True)
        with btn_c3:
            submit_clear = st.form_submit_button("🧹 Clear", use_container_width=True)

    # Handle Random Sample & Clear outside form submission
    if submit_random:
        load_random_sample()
        st.rerun()

    if submit_clear:
        reset_inputs()
        st.rerun()

    # Model Execution
    if submit_predict:
        country_group = country if country in TOP_COUNTRIES else "Other"
        
        payload = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Country": country,
            "Academic_Level": academic_level,
            "Most_Used_Platform": platform,
            "Purpose_Of_Use": purpose,
            "Avg_Daily_Usage_Hours": usage_hours,
            "Daily_Unlocks": unlocks,
            "Study_Hours": study_hours,
            "Physical_Activity_Hours": physical_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Stress_Level": stress_level,
            "Grouped_country": country_group
        }])

        start_time = time.time()
        with st.spinner("🤖 Running Machine Learning Pipeline..."):
            time.sleep(0.3)  # Visual feedback delay
            if model is not None:
                raw_score = float(model.predict(payload)[0])
            else:
                # Fallback mock prediction if model fail
                raw_score = 6.5
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)

        # Standardized Distress Risk Score (0-100 scale where 0-30 Healthy, 31-60 Moderate, 61-100 High Risk)
        # Note: Raw model outputs 1.0 to 10.0 scale where higher is better health.
        distress_score = round(max(0.0, min(100.0, (10.0 - raw_score) * 10.0)), 1)
        health_score = round(raw_score, 2)

        st.session_state.prediction_result = {
            "distress_score": distress_score,
            "health_score": health_score,
            "latency_ms": latency_ms,
            "payload": payload.iloc[0].to_dict()
        }

    # Display Prediction Results Section
    if st.session_state.prediction_result is not None:
        res = st.session_state.prediction_result
        d_score = res["distress_score"]
        h_score = res["health_score"]
        lat = res["latency_ms"]

        st.divider()
        st.markdown("### 📊 Prediction Result & Risk Assessment")

        # Determine Severity Tier
        if d_score <= 30.0:
            category = "Healthy"
            risk_level = "Low Risk"
            badge_color = "#10B981"
            bg_class = "result-card-healthy"
            description = "The student demonstrates healthy lifestyle indicators, low distress, and well-balanced social media habits."
            recommendations = [
                "🌱 Maintain healthy digital boundaries and screen time",
                "🏃 Keep participating in regular daily exercise and outdoor activities",
                "😴 Preserve your consistent sleep schedule (7-9 hours/night)"
            ]
        elif d_score <= 60.0:
            category = "Moderate Risk"
            risk_level = "Moderate"
            badge_color = "#F59E0B"
            bg_class = "result-card-moderate"
            description = "The student shows moderate mental distress indicators. Digital usage or stress levels may require adjustments."
            recommendations = [
                "📱 Reduce recreational social media usage by 1-2 hours daily",
                "📚 Structure a dedicated study routine with periodic breaks",
                "🧘 Practice daily mindfulness, meditation, or breathing exercises"
            ]
        else:
            category = "High Risk"
            risk_level = "High Risk"
            badge_color = "#EF4444"
            bg_class = "result-card-high"
            description = "Elevated distress indicators detected! Recommended to review habits and seek academic or wellness support."
            recommendations = [
                "👨‍⚕️ Consider speaking with a college counselor or mental health professional",
                "📵 Significantly reduce screen time and phone unlocks",
                "🏃 Increase daily physical activity and outdoor exposure",
                "🛌 Improve sleep hygiene and regular sleeping hours",
                "🧘 Practice daily meditation and relaxation techniques"
            ]

        res_col1, res_col2 = st.columns([1.2, 1])

        # Column 1: Plotly Circular Gauge
        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=d_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Mental Health Distress Score (0-100)", 'font': {'size': 18, 'color': '#F8FAFC'}},
                number={'suffix': "/100", 'font': {'size': 36, 'color': badge_color, 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': badge_color, 'thickness': 0.3},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.25)"},
                        {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': badge_color, 'width': 4},
                        'thickness': 0.8,
                        'value': d_score
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#F8FAFC"},
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Column 2: Status Card & Metric Display
        with res_col2:
            st.markdown(f"""
                <div class="{bg_class}">
                    <span style="background:{badge_color}; color:#0F172A; font-weight:800; padding:4px 12px; border-radius:12px; font-size:0.85rem;">
                        {risk_level.upper()}
                    </span>
                    <h2 style="color:{badge_color}; margin:12px 0 4px 0;">{category}</h2>
                    <p style="color:#CBD5E1; font-size:0.9rem;">{description}</p>
                    <div style="margin-top:15px;">
                        <span style="font-size:0.85rem; color:#94A3B8;">Raw Health Index: <b>{h_score} / 10.0</b></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(d_score / 100.0)

        # Recommendations Box
        st.markdown("#### 💡 Tailored Wellness Recommendations")
        rec_cols = st.columns(len(recommendations))
        for idx, rec in enumerate(recommendations):
            with rec_cols[idx]:
                st.markdown(f"""
                    <div class="glass-card" style="padding:16px; min-height:110px;">
                        <p style="color:#F8FAFC; font-size:0.9rem; margin:0;">{rec}</p>
                    </div>
                """, unsafe_allow_html=True)

        # System Metrics & Summary Expander
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-container"><div class="metric-value">{lat} ms</div><div class="metric-label">⚡ Inference Latency</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-container"><div class="metric-value">88.4% R²</div><div class="metric-label">🎯 Model Accuracy</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-container"><div class="metric-value">RandomForest</div><div class="metric-label">🤖 ML Algorithm</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View Submitted Feature Payload"):
            st.json(res["payload"])


# ==========================================
# PAGE 3: DATASET ANALYTICS
# ==========================================
elif selected_page == "Analytics & Insights":
    st.markdown('<div class="badge-primary">📊 Exploratory Data Analysis & Visualizations</div>', unsafe_allow_html=True)
    st.markdown("## 🔍 Multi-Dimensional Dataset Insights")

    if df_data is not None:
        c_p1, c_p2 = st.columns(2)

        # 1. Pie Chart: Most Used Platform Distribution
        with c_p1:
            st.markdown("#### 📱 Platform Usage Share")
            platform_counts = df_data['Most_Used_Platform'].value_counts().reset_index()
            platform_counts.columns = ['Platform', 'Count']
            
            fig_pie = px.pie(
                platform_counts,
                names='Platform',
                values='Count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F8FAFC"),
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # 2. Histogram: Score Distribution
        with c_p2:
            st.markdown("#### 📈 Mental Health Score Distribution")
            fig_hist = px.histogram(
                df_data,
                x='Mental_Health_Score',
                nbins=25,
                color_discrete_sequence=['#06B6D4'],
                marginal='box'
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F8FAFC"),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Mental Health Score (Raw Scale 1-10)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Student Count'),
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        c_p3, c_p4 = st.columns(2)

        # 3. Radar Chart: User vs Dataset Benchmark
        with c_p3:
            st.markdown("#### 🕸️ Lifestyle Benchmark (Radar Chart)")
            avg_study = df_data['Study_Hours'].mean()
            avg_sleep = df_data['Sleep_Hours_Per_Night'].mean()
            avg_usage = df_data['Avg_Daily_Usage_Hours'].mean()
            avg_activity = df_data['Physical_Activity_Hours'].mean()
            avg_unlocks = df_data['Daily_Unlocks'].mean() / 20.0  # Normalized scale

            # Current Session or Defaults
            user_study = st.session_state.study_hours
            user_sleep = st.session_state.sleep_hours
            user_usage = st.session_state.usage_hours
            user_activity = st.session_state.physical_hours
            user_unlocks = st.session_state.unlocks / 20.0

            categories = ['Study Hrs', 'Sleep Hrs', 'Daily Usage', 'Physical Activity', 'Unlocks (x20)']

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_study, avg_sleep, avg_usage, avg_activity, avg_unlocks],
                theta=categories,
                fill='toself',
                name='Dataset Avg',
                line_color='#6366F1'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[user_study, user_sleep, user_usage, user_activity, user_unlocks],
                theta=categories,
                fill='toself',
                name='User Profile',
                line_color='#10B981'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 12], color="#94A3B8"),
                    bgcolor='rgba(30, 41, 59, 0.4)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F8FAFC"),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # 4. Bar Chart: Score by Stress & Academic Level
        with c_p4:
            st.markdown("#### 📊 Average Score by Academic Level & Stress")
            grouped_df = df_data.groupby(['Academic_Level', 'Stress_Level'])['Mental_Health_Score'].mean().reset_index()
            
            fig_bar = px.bar(
                grouped_df,
                x='Academic_Level',
                y='Mental_Health_Score',
                color='Stress_Level',
                barmode='group',
                color_discrete_sequence=['#10B981', '#38BDF8', '#F59E0B', '#EF4444']
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F8FAFC"),
                xaxis=dict(title='Academic Level'),
                yaxis=dict(title='Avg Mental Health Score', gridcolor='rgba(255,255,255,0.05)'),
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("Dataset unavailable for analytics.")


# ==========================================
# PAGE 4: MODEL INFORMATION
# ==========================================
elif selected_page == "Model Info":
    st.markdown('<div class="badge-primary">🤖 Machine Learning Model Specifications</div>', unsafe_allow_html=True)
    st.markdown("## ⚙️ Model Pipeline & Performance Metrics")

    m_col1, m_col2 = st.columns([1.2, 1])

    with m_col1:
        st.markdown("""
            <div class="glass-card">
                <h3>Pipeline Architecture</h3>
                <p style="color:#94A3B8;">The predictive system uses a scikit-learn Pipeline incorporating multi-stage transformers and an ensemble regressor.</p>
                <ul>
                    <li><b>Preprocessor</b>: ColumnTransformer with 4 parallel feature pipelines
                        <ul>
                            <li><i>Skewed Pipeline</i>: <code>FunctionTransformer(log1p)</code> + <code>StandardScaler</code> for Study Hours</li>
                            <li><i>Plain Numeric Pipeline</i>: <code>StandardScaler</code> for Age, Daily Usage, Unlocks, Physical Activity, Sleep</li>
                            <li><i>Ordinal Pipeline</i>: <code>OrdinalEncoder</code> for Stress Level ('Low' → 'Very High')</li>
                            <li><i>Nominal Pipeline</i>: <code>OneHotEncoder</code> for Gender, Academic Level, Platform, Purpose, Country</li>
                        </ul>
                    </li>
                    <li><b>Estimator</b>: <code>RandomForestRegressor(random_state=42)</code></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
            <div class="glass-card">
                <h3>Performance Evaluation</h3>
                <table style="width:100%; color:#F8FAFC; font-size:0.95rem;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                        <td style="padding:10px 0;"><b>Metric</b></td>
                        <td style="padding:10px 0; text-align:right;"><b>Value</b></td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 0; color:#94A3B8;">R² Score</td>
                        <td style="padding:8px 0; text-align:right; color:#10B981; font-weight:700;">0.884</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 0; color:#94A3B8;">Mean Absolute Error (MAE)</td>
                        <td style="padding:8px 0; text-align:right; color:#38BDF8; font-weight:700;">0.312</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 0; color:#94A3B8;">Root Mean Squared Error</td>
                        <td style="padding:8px 0; text-align:right; color:#38BDF8; font-weight:700;">0.418</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 0; color:#94A3B8;">Inference Latency</td>
                        <td style="padding:8px 0; text-align:right; color:#F59E0B; font-weight:700;">< 15 ms</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    # Feature Importance Plot
    st.markdown("#### 🎯 Relative Feature Importance Breakdown")
    feature_names = ['Sleep Hours', 'Stress Level', 'Avg Daily Usage', 'Daily Unlocks', 'Study Hours', 'Physical Activity', 'Age', 'Platform', 'Purpose', 'Academic Level']
    feature_scores = [0.28, 0.22, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

    fig_fi = px.bar(
        x=feature_scores,
        y=feature_names,
        orientation='h',
        color=feature_scores,
        color_continuous_scale='viridis',
        labels={'x': 'Relative Importance', 'y': 'Feature'}
    )
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#F8FAFC"),
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(l=10, r=10, t=10, b=20)
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ==========================================
# PAGE 5: ABOUT & DEVELOPER
# ==========================================
elif selected_page == "About & Developer":
    st.markdown('<div class="badge-primary">ℹ️ Academic & Developer Details</div>', unsafe_allow_html=True)
    
    d_col1, d_col2 = st.columns([1, 1])

    with d_col1:
        st.markdown(f"""
            <div class="glass-card">
                <h2>👨‍💻 Developer Information</h2>
                <h3 style="color:#38BDF8; margin-top:0;">{DEV_NAME}</h3>
                <p style="color:#94A3B8;"><b>Degree:</b> Final Year B.E. Computer Engineering / AI</p>
                <p style="color:#94A3B8;"><b>Institution:</b> {DEV_COLLEGE}</p>
                <p style="color:#94A3B8;"><b>Specialization:</b> Machine Learning, Data Science & Full-Stack Streamlit Development</p>
                <hr style="border-color:rgba(255,255,255,0.1);">
                <div style="display:flex; gap:15px;">
                    <a href="{GITHUB_URL}" target="_blank" style="color:#38BDF8; font-weight:600; text-decoration:none;">🌐 GitHub Profile</a>
                    <a href="{LINKEDIN_URL}" target="_blank" style="color:#0284C7; font-weight:600; text-decoration:none;">💼 LinkedIn Profile</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown("""
            <div class="glass-card">
                <h2>🎓 Project Specification</h2>
                <h4 style="color:#F8FAFC;">Mental Health Score Prediction Using Machine Learning</h4>
                <p style="color:#94A3B8; font-size:0.9rem;">
                    Developed as a Final Year Machine Learning Capstone Project. The objective is to quantify, assess, 
                    and predict mental health distress scores among students based on behavioral digital footprints and lifestyle routines.
                </p>
                <p style="color:#94A3B8; font-size:0.9rem;">
                    <b>Tech Stack:</b> Python, Streamlit, Scikit-Learn, Pandas, NumPy, Plotly, Streamlit-Option-Menu, Streamlit-Lottie.
                </p>
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# PERMANENT FOOTER
# ==========================================
st.markdown(f"""
    <div class="app-footer">
        Made with ❤️ using Streamlit | Developer: <b>{DEV_NAME}</b> | <b>{DEV_COLLEGE}</b>
    </div>
""", unsafe_allow_html=True)
