# 🧠 MindPulse AI — Student Mental Health Score Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Render Deploy](https://img.shields.io/badge/Deploy%20to-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end Machine Learning web application and API predicting student mental health risk scores based on social media usage, academic habits, physical activity, and sleep patterns.

---

## 🌟 Key Features

- 🎯 **Accurate ML Prediction**: Random Forest Regressor & Gradient Boosting Pipeline trained on comprehensive student behavioral data.
- 🎨 **Modern Glassmorphic Dark UI**: Built with Streamlit, custom CSS, Plotly interactive charts, and responsive cards.
- 📊 **Interactive Analytics Dashboard**: Deep-dive exploratory data analysis on social media platforms, sleep vs. screen time, and stress metrics.
- ⚡ **Dual Engine**:
  - **Streamlit Web Application** (`streamlit_app.py`): Full UI for interactive prediction and exploratory data analytics.
  - **FastAPI REST Service** (`main.py`): High-speed JSON endpoint (`/predict`) for programmatic integration.
- 🚀 **Cloud Ready**: Configured with `render.yaml`, `Procfile`, and `.streamlit/config.toml` for seamless Render deployment.

---

## 🏗️ Project Architecture

```
Student Mental Health Score Predictor/
├── .streamlit/
│   └── config.toml                  # Streamlit cloud and dark theme config
├── Mental_Health_Model.pkl          # Trained Scikit-Learn Pipeline Model
├── Student Health Score.ipynb       # Jupyter Notebook (EDA, training, evaluation)
├── Student Social Media And Mental Health Impact.csv # Dataset
├── main.py                          # FastAPI REST API Backend
├── streamlit_app.py                 # Streamlit Web App Interface
├── render.yaml                      # Render Blueprint configuration
├── Procfile                         # Cloud process runner command
├── requirements.txt                 # Production dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Documentation
```

---

## 🚀 Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vivekpatil77/Student-Mental-Health-Score-Predictor.git
cd Student-Mental-Health-Score-Predictor
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 5. (Optional) Run the FastAPI Service
```bash
uvicorn main:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

---

## 🌐 Deploy to Render (Step-by-Step Guide)

Deploying this app on **Render** is completely free and takes only a few minutes:

### Method 1: Automatic Blueprint (Recommended)
1. Sign in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repository (`Student-Mental-Health-Score-Predictor`).
4. Render will automatically detect `render.yaml` and configure:
   - **Environment**: `Python`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
5. Click **Apply**. Render will build and launch your live web service!

### Method 2: Manual Web Service Creation
1. In [Render Dashboard](https://dashboard.render.com/), click **New +** → **Web Service**.
2. Select **Build and deploy from a Git repository** and connect your GitHub repo.
3. Configure the following settings:
   - **Name**: `student-mental-health-predictor`
   - **Language / Environment**: `Python 3`
   - **Branch**: `main` (or `master`)
   - **Region**: Closest to you (e.g., `Singapore` or `Oregon`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
   - **Instance Type**: `Free`
4. Under **Advanced** → **Add Environment Variable**:
   - `PYTHON_VERSION`: `3.11.9`
5. Click **Create Web Service**.

---

## 📡 REST API Usage (FastAPI)

### Predict Endpoint (`POST /predict`)
```bash
curl -X POST "https://your-render-url.onrender.com/predict" \
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

**Response**:
```json
{
  "predicted_mental_health_score": 6.42
}
```

---

## 👨‍💻 Author

- **Vivek Namdev Patil**
- **College**: AISSMS College of Engineering Pune
- **GitHub**: [@Vivekpatil77](https://github.com/Vivekpatil77)
- **LinkedIn**: [Vivek Patil](https://linkedin.com)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
