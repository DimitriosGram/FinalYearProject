import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

###############################################
# 1) Load & Encode Match-Level Data
###############################################
def load_match_data(csv_path: str) -> pd.DataFrame:
    """Loads the 11k+ row match-level CSV and processes key features."""
    df = pd.read_csv(csv_path, encoding='latin-1')
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

    # Parse 'Season' (e.g., "2018-19") -> 2018
    def parse_season_start(s):
        try:
            return int(s.split('-')[0])
        except:
            return np.nan

    df['SeasonStart'] = df['Season'].apply(parse_season_start)
    df.dropna(subset=['SeasonStart'], inplace=True)
    df['SeasonStart'] = df['SeasonStart'].astype(int)

    return df

def encode_ftr(df: pd.DataFrame) -> pd.DataFrame:
    """Maps FTR (match result) to numeric values."""
    mapping = {'H':2, 'D':1, 'A':0}
    df['MatchResult'] = df['FTR'].map(mapping)
    return df

def time_based_split(df: pd.DataFrame, start_year=2010, split_year=2020):
    """Filters data and applies a time-based split for training/testing."""
    df = df[df['SeasonStart'] >= start_year]
    train_df = df[df['SeasonStart'] < split_year]
    test_df = df[df['SeasonStart'] >= split_year]
    return train_df, test_df

###############################################
# 2) Train Match Outcome Classifier
###############################################
def train_outcome_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Trains an XGBoost classifier to predict match results."""
    feats = ['HS','HST','HC','AS','AST','AC']
    label = 'MatchResult'

    train_df = train_df.dropna(subset=feats + [label])
    test_df = test_df.dropna(subset=feats + [label])

    X_train = train_df[feats].astype(float)
    y_train = train_df[label].astype(int)
    X_test = test_df[feats].astype(float)
    y_test = test_df[label].astype(int)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, acc, (X_test_sc, y_test)

###############################################
# 3) Team Average Home/Away Stats
###############################################
def build_team_averages(df: pd.DataFrame):
    """Computes average home/away performance for each team."""
    home_agg = df.groupby('HomeTeam', as_index=False)[['HS','HST','HC']].mean()
    home_agg.rename(columns={'HomeTeam':'Team'}, inplace=True)
    home_agg.columns = ['Team','home_HS','home_HST','home_HC']

    away_agg = df.groupby('AwayTeam', as_index=False)[['AS','AST','AC']].mean()
    away_agg.rename(columns={'AwayTeam':'Team'}, inplace=True)
    away_agg.columns = ['Team','away_AS','away_AST','away_AC']

    merged = pd.merge(home_agg, away_agg, on='Team', how='outer')
    return merged

###############################################
# 4) Simulating Future Seasons (38 Matches)
###############################################
def create_demo_fixture_list(teams, n_matches=38):
    """Creates a randomized fixture list for a full season."""
    rng = np.random.default_rng(seed=42)
    fixtures = []
    n_teams = len(teams)
    
    for i in range(n_matches):
        home_idx = rng.integers(0, n_teams)
        away_idx = rng.integers(0, n_teams)
        while away_idx == home_idx:
            away_idx = rng.integers(0, n_teams)
        
        fixtures.append({"HomeTeam": teams[home_idx], "AwayTeam": teams[away_idx]})
    
    return pd.DataFrame(fixtures)

def simulate_season_winner(model, scaler, team_avgs, teams, season_label):
    """Simulates a season and determines the champion based on most wins."""
    fixtures_df = create_demo_fixture_list(teams, n_matches=38)
    wins_counter = {t:0 for t in teams}

    for _, row in fixtures_df.iterrows():
        home_t = row['HomeTeam']
        away_t = row['AwayTeam']

        row_home = team_avgs[team_avgs['Team'] == home_t]
        row_away = team_avgs[team_avgs['Team'] == away_t]
        
        if row_home.empty or row_away.empty:
            continue

        # Build feature row
        feat = pd.DataFrame([[
            row_home['home_HS'].values[0],
            row_home['home_HST'].values[0],
            row_home['home_HC'].values[0],
            row_away['away_AS'].values[0],
            row_away['away_AST'].values[0],
            row_away['away_AC'].values[0]
        ]], columns=['HS','HST','HC','AS','AST','AC'])

        feat_sc = scaler.transform(feat)
        pred_outcome = model.predict(feat_sc)[0]

        if pred_outcome == 2:
            wins_counter[home_t] += 1
        elif pred_outcome == 0:
            wins_counter[away_t] += 1

    champion = max(wins_counter, key=wins_counter.get)
    return champion

def predict_next_10_years_winners(model, scaler, team_avgs, recent_teams):
    """Simulates the next 10 seasons and predicts winners."""
    future_results = []
    start_year = 2024
    
    for i in range(10):
        season_str = f"{start_year+i}-{(start_year+i+1)%100:02d}"
        champ = simulate_season_winner(model, scaler, team_avgs, recent_teams, season_str)
        future_results.append({"Season": season_str, "Champion": champ})

    return pd.DataFrame(future_results)

###############################################
# 5) Streamlit App
###############################################
def main():
    st.title("Match Outcome Predictor & Future Champions")

    # Load & process match data
    df = load_match_data("Team Stats/results.csv")
    df = encode_ftr(df)

    # Train-test split
    train_df, test_df = time_based_split(df, start_year=2005, split_year=2019)
    st.write(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    # Train classifier
    model, scaler, acc, (X_test_sc, y_test) = train_outcome_classifier(train_df, test_df)
    st.write(f"Accuracy on test set: {acc:.3f}")

    # Compute team averages
    team_avgs = build_team_averages(df)
    recent_teams = team_avgs['Team'].dropna().unique().tolist()

    # Dropdown menu
    user_choice = st.selectbox("Choose an option:", [
        "View test predictions",
        "Predict next 10 years winners"
    ])

    if user_choice == "View test predictions":
        sample = test_df.sample(10, random_state=42).copy()
        feats = ['HS','HST','HC','AS','AST','AC']
        sample.dropna(subset=feats + ['MatchResult'], inplace=True)
        scaler_features = scaler.transform(sample[feats])
        preds = model.predict(scaler_features)
        inv_map = {2:'H', 1:'D', 0:'A'}
        sample['PredictedFTR'] = [inv_map[p] for p in preds]
        st.dataframe(sample[['Season','DateTime','HomeTeam','AwayTeam','FTR','PredictedFTR']])

    elif user_choice == "Predict next 10 years winners":
        future_df = predict_next_10_years_winners(model, scaler, team_avgs, recent_teams)
        st.write("**Predicted Champions** for the next 10 years:")
        st.dataframe(future_df)

if __name__ == "__main__":
    main()