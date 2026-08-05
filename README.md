<div align="center">

# 🧠 MindPulse AI
### Student Mental Health Score Predictor & Analytics Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Deploy to Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>An AI-powered diagnostic platform combining Machine Learning, Interactive Visualizations, and RESTful APIs to predict and analyze student mental health risk scores.</b>
</p>

</div>

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [📊 Feature Attributes & Input Schema](#-feature-attributes--input-schema)
- [🎯 Mental Health Score Interpretation](#-mental-health-score-interpretation)
- [🔬 Machine Learning Pipeline](#-machine-learning-pipeline)
- [🏗️ Project Structure](#️-project-structure)
- [💻 Local Installation & Setup](#-local-installation--setup)
- [🚀 1-Click Deployment to Render](#-1-click-deployment-to-render)
- [📡 FastAPI REST API Documentation](#-fastapi-rest-api-documentation)
- [👨‍💻 Author & Contact](#-author--contact)
- [📄 License](#-license)

---

## ✨ Key Features

- 🎯 **Accurate Risk Scoring**: Uses an ensemble ML Pipeline trained on real student lifestyle, academic, and behavioral metrics.
- 🎨 **Modern Glassmorphic Dark UI**: Built with Streamlit, custom CSS glassmorphic cards, gradient accents, and micro-animations.
- 📈 **Interactive Exploratory Data Analysis**:
  - Social media platform distribution & usage breakdown.
  - Sleep duration vs. Screen time correlation analysis.
  - Stress level distributions across academic tiers.
- ⚡ **Dual Engine Architecture**:
  - **Streamlit Web Application** (`streamlit_app.py`): Full UI with real-time sliders, visual gauges, and personalized recommendations.
  - **FastAPI REST Endpoint** (`main.py`): High-throughput `/predict` endpoint for webhooks, mobile apps, and third-party integrations.
- ☁️ **Cloud Native**: Pre-configured with `render.yaml`, `Procfile`, and `.streamlit/config.toml` for seamless 1-click deployment.

---

## 📊 Feature Attributes & Input Schema

| Feature Name | Type | Range / Options | Description |
| :--- | :--- | :--- | :--- |
| `Age` | Numerical | 10 – 100 years | Student's current age |
| `Gender` | Categorical | `Male`, `Female` | Biological gender |
| `Country` | Categorical | Top 10 countries / `Other` | Geographical location |
| `Academic_Level` | Categorical | `High School`, `Undergraduate`, `Graduate` | Current level of study |
| `Most_Used_Platform` | Categorical | `Instagram`, `YouTube`, `TikTok`, `Snapchat`, `Facebook`, `LinkedIn`, `WhatsApp`, `Twitter`, etc. | Primary social media platform |
| `Purpose_Of_Use` | Categorical | `Networking`, `Education`, `Entertainment`, `News` | Primary reason for social media usage |
| `Avg_Daily_Usage_Hours` | Numerical | 0.0 – 24.0 hrs/day | Screen time spent on social media |
| `Daily_Unlocks` | Numerical | 0 – 500+ times/day | Smartphone pickup / unlock frequency |
| `Study_Hours` | Numerical | 0.0 – 24.0 hrs/day | Self-study / academic hours |
| `Physical_Activity_Hours`| Numerical | 0.0 – 24.0 hrs/day | Daily sports / exercise time |
| `Sleep_Hours_Per_Night` | Numerical | 0.0 – 24.0 hrs/night | Average night sleep duration |
| `Stress_Level` | Categorical | `Low`, `Medium`, `High`, `Very High` | Self-reported stress level |

---

## 🎯 Mental Health Score Interpretation

| Predicted Score | Risk Level | Status | Recommended Action |
| :---: | :---: | :---: | :--- |
| **0.0 – 3.9** | 🟢 Low Risk | **Healthy & Balanced** | Maintain current routine, sleep schedule, and physical activity. |
| **4.0 – 6.9** | 🟡 Moderate Risk | **Attention Advised** | Reduce screen time/unlocks, increase physical exercise and study breaks. |
| **7.0 – 10.0** | 🔴 High Risk | **Action Required** | Consult academic counselor or healthcare professional; practice digital detox. |

---

## 🔬 Machine Learning Pipeline

The ML pipeline is saved in `Mental_Health_Model.pkl` and consists of:
1. **Preprocessor (ColumnTransformer)**:
   - `StandardScaler` for continuous numerical features.
   - `OrdinalEncoder` for ordinal levels (e.g., Stress Level, Academic Level).
   - `OneHotEncoder` for nominal categorical variables (Gender, Platform, Country).
2. **Regressor**:
   - Tuned `RandomForestRegressor` with cross-validation on student mental health indicators.

---

## 🏗️ Project Structure

```
Student Mental Health Score Predictor/
├── .streamlit/
│   └── config.toml                  # Cloud server & dark glassmorphic theme configuration
├── Mental_Health_Model.pkl          # Trained Scikit-Learn Pipeline
├── Student Health Score.ipynb       # Jupyter Notebook (EDA, Modeling, Evaluation)
├── Student Social Media And Mental Health Impact.csv # Dataset
├── main.py                          # FastAPI REST API Backend
├── streamlit_app.py                 # Streamlit Interactive Web Application
├── render.yaml                      # Render Blueprint (Infrastructure-as-Code)
├── Procfile                         # Process declaration for cloud platforms
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Git ignore list
└── README.md                        # Documentation & guides
```

---

## 💻 Local Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vivekpatil77/Student-Mental-Health-Score-Predictor.git
cd Student-Mental-Health-Score-Predictor
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 5. Run the FastAPI Backend (Optional)
```bash
uvicorn main:app --reload --port 8000
```
Interactive Swagger API docs will be live at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🚀 1-Click Deployment to Render

### Method 1: Blueprint Deployment (Recommended)
1. Fork or push this repository to your GitHub account.
2. Sign in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Select `Student-Mental-Health-Score-Predictor`.
5. Render reads [`render.yaml`](render.yaml) automatically. Click **Apply** to deploy!

### Method 2: Manual Web Service
1. Click **New +** → **Web Service** → Connect your GitHub repo.
2. Enter the following parameters:
   - **Environment**: `Python 3`
   - **Region**: Any (e.g. `Singapore` or `Oregon`)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**:
     ```bash
     streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
     ```
   - **Environment Variable**: `PYTHON_VERSION` = `3.11.9`
3. Click **Create Web Service**.

---

## 📡 FastAPI REST API Documentation

### Predict Endpoint (`POST /predict`)

#### Request Body
```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 120,
  "study_hours": 3.5,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 6.5,
  "stress_level": "Medium"
}
```

#### Example cURL Request
```bash
curl -X POST "https://your-app.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 21,
       "gender": "Male",
       "country": "India",
       "academic_level": "Undergraduate",
       "most_used_platform": "Instagram",
       "purpose_of_use": "Entertainment",
       "avg_daily_usage_hours": 4.5,
       "daily_unlocks": 120,
       "study_hours": 3.5,
       "physical_activity_hours": 1.0,
       "sleep_hours_per_night": 6.5,
       "stress_level": "Medium"
     }'
```

#### Response
```json
{
  "predicted_mental_health_score": 6.42
}
```

---

## 👨‍💻 Author & Contact

- **Author**: Vivek Namdev Patil
- **College**: AISSMS College of Engineering Pune
- **GitHub**: [@Vivekpatil77](https://github.com/Vivekpatil77)
- **LinkedIn**: [Vivek Patil](https://linkedin.com)

---

## 📄 License
This project is open-source and licensed under the **[MIT License](LICENSE)**.
