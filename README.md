# ⚽ Premier League Predictor

An intelligent **football analysis and forecasting system** that predicts Premier League outcomes using **machine learning, statistical modeling**, and **psychological factors** influencing team performance.
Developed as part of a **Final Year Project (BSc Computer Science)**, this tool integrates historical data, team form, discipline metrics, and mental pressure indicators to generate accurate end-of-season predictions.

---

## 🚀 Key Features

* Predicts **final league standings** based on current and historical season data.
* Incorporates **psychological and discipline metrics** such as:

  * Yellow/red card trends
  * Foul rates and match discipline
  * Team pressure status (e.g., relegation battles, title contention)
  * Recent form under high-pressure conditions
* Uses **rolling performance windows** to reflect team momentum and confidence.
* Interactive **Streamlit web interface** with dynamic cutoff sliders to simulate season progress.
* Visual dashboards showing:

  * Final points predictions
  * Over 2.5 Goals probabilities
  * Expected Corners
  * Home vs Away Expected Goals

---

## 🧠 Technical Overview

This project combines machine learning with psychological and behavioral analysis to produce context-aware forecasts. Traditional sports predictors focus purely on numerical features like goals or wins. This project expands that by integrating mental and emotional aspects such as player discipline, pressure status, and recent form stability, which often influence real-world results but are rarely modeled quantitatively.

---

### ⚙️ System Architecture

```text
+--------------------+
|  FBref Scraper     |  --> Extracts player/team data (discipline, form, stats)
+---------+----------+
          |
          v
+--------------------+
|  AWS S3 Storage    |  --> Stores structured season datasets
+---------+----------+
          |
          v
+--------------------+
|  ProjectAlpha.py   |  --> ML model + Streamlit interface
|  (XGBoost Engine)  |
+---------+----------+
          |
          v
+--------------------+
|  Prediction Output |  --> Points, discipline risk, form analytics
+--------------------+
```

### 🧹 Components

| Component               | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `ProjectAlpha.py`       | Core Streamlit app integrating the machine learning pipeline.                 |
| `S3 Datasets`           | Season-level and match-level data including historical and current seasons.   |
| `Psychological Metrics` | Derived features capturing discipline, pressure status, and momentum.         |
| `XGBoost Regressor`     | Trained model for predicting final season points.                             |
| `Streamlit UI`          | Interactive frontend for visualizing results and adjusting cut-off scenarios. |

### 📊 Model Inputs

| Category      | Example Features                                                                  |
| ------------- | --------------------------------------------------------------------------------- |
| Performance   | Wins, losses, goals for/against, expected goals, form (last 5 matches)            |
| Psychological | Pressure indicator (relegation/title contention), match importance, stress factor |
| Discipline    | Yellow/red cards per game, fouls per match, aggression index                      |
| External      | Home/away status, opponent form, referee profile (optional future feature)        |

These engineered features ensure that the model doesn’t just "see" results — it interprets the conditions under which results occur.

---

## 🛠️ Installation

To run the project locally:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install streamlit pandas numpy scikit-learn xgboost boto3
```

If using AWS S3 for data storage:

```bash
aws configure
```

Ensure your S3 paths follow the expected structure:

```bash
teamData/Results/Results
teamData/Results/PastResults
```

These paths contain both the match-level and season-level datasets used for model training and predictions.

---

## ▶️ Running the App

```bash
streamlit run NewModel.py
```

Once launched, the web interface will:

* Load the latest match and team data from AWS S3
* Predict and visualize final league standings
* Display charts for goals, corners, and psychological risk factors

The app uses Streamlit’s interactive components (sliders, charts, and tables) to simulate different points in the season, showing how predictions evolve with new data.

---

## 🔮 Future Development

Planned enhancements aim to expand the psychological modeling framework and automate real-time data ingestion:

* Integrate a dedicated Discipline & Psychology Risk Model predicting:

  * Remaining yellow/red card likelihoods
  * Team composure under pressure
  * Impact of discipline on future results

* Expand psychological profiling to include:

  * Momentum streaks
  * Media/fan pressure (quantitative proxies)
  * Coaching stability and leadership indicators

* Create an API + dashboard layer for real-time updates and automated predictions.

This direction turns the system into a full sports intelligence platform — combining statistical accuracy with behavioral understanding.

---

## 🗃️ Academic Context

This project investigates how psychological and behavioral dynamics influence statistical football forecasting.
By combining data science, AI, and sports psychology, it offers a hybrid analytical framework that moves beyond pure numerical modeling toward context-aware prediction.

### Key Domains

* Machine Learning & Predictive Analytics
* Sports Data Engineering (AWS S3, Lambda, Selenium)
* Human Factors in AI
* Statistical Feature Engineering

This multidisciplinary design demonstrates how psychological dimensions can enhance the reliability and realism of football forecasting models, reflecting real-world influences beyond raw performance statistics.

---

## 🧑‍💻 Author

**Dimitrios**
Final Year Computer Science Student
Passionate about applied AI, data engineering, and sports intelligence systems.
Focused on bridging machine learning and human behavioral analysis to improve predictive accuracy in competitive environments.

---

## 📜 License

This project is for academic and research purposes only.
**Unauthorized commercial use or redistribution is prohibited.**
