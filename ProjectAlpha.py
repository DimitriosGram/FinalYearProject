import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------------------
# 1) Build Final Table (Full-Season) - with GF, GA, GD
# -----------------------------------------------------
def build_final_table(csv_path: str) -> pd.DataFrame:
    """
    Reads the match-level CSV (1993-2022) and aggregates
    to get final seasonal totals (Points, GF, GA, Shots, etc.)
    for each (Season, Team).
    Adds a 'GD' = GF - GA.
    """
    df = pd.read_csv(csv_path, encoding="latin-1")
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    # Rename "Pts" -> "Points" if it exists
    if "Pts" in df.columns:
        df.rename(columns={"Pts": "Points"}, inplace=True)

    # Create home/away points from FTR
    def get_points_for_home_team(result):
        return 3 if result == 'H' else 1 if result == 'D' else 0
    def get_points_for_away_team(result):
        return 3 if result == 'A' else 1 if result == 'D' else 0

    df['HomePoints'] = df['FTR'].apply(get_points_for_home_team)
    df['AwayPoints'] = df['FTR'].apply(get_points_for_away_team)

    # HOME aggregator
    home_agg = df.groupby(['Season', 'HomeTeam'], as_index=False).agg({
        'HomePoints': 'sum',
        'FTHG': 'sum',  # goals for
        'FTAG': 'sum',  # goals against
        'HS': 'sum',
        'HST': 'sum',
        'HC': 'sum',
    })
    home_agg.rename(columns={'HomeTeam': 'Team', 'HomePoints': 'Points'}, inplace=True)

    # AWAY aggregator
    away_agg = df.groupby(['Season', 'AwayTeam'], as_index=False).agg({
        'AwayPoints': 'sum',
        'FTAG': 'sum',  # away goals for
        'FTHG': 'sum',  # home goals against
        'AS': 'sum',
        'AST': 'sum',
        'AC': 'sum',
    })
    away_agg.rename(columns={'AwayTeam': 'Team', 'AwayPoints': 'Points'}, inplace=True)

    # Combine
    combined = pd.concat([home_agg, away_agg], ignore_index=True)

    final = combined.groupby(['Season','Team'], as_index=False).agg({
        'Points': 'sum',
        'FTHG': 'sum',  # total goals for
        'FTAG': 'sum',  # total goals against
        'HS': 'sum',
        'HST': 'sum',
        'HC': 'sum',
    })

    # Rename FTHG->GF, FTAG->GA, add GD
    final.rename(columns={
        'FTHG': 'GF',
        'FTAG': 'GA'
    }, inplace=True)
    final['GD'] = final['GF'] - final['GA']

    print("Full-season aggregator columns:", final.columns.tolist())
    return final


# -----------------------------------------------------
# 2) Build Partial-Season Dataset
# -----------------------------------------------------
def build_partial_dataset(all_matches_df: pd.DataFrame, final_table: pd.DataFrame, N=24) -> pd.DataFrame:
    rows = []
    for idx, row in all_matches_df.iterrows():
        season = row['Season']
        date   = row['DateTime']
        home_team = row['HomeTeam']
        home_result = row['FTR']
        if home_result == 'H':
            home_points = 3
        elif home_result == 'D':
            home_points = 1
        else:
            home_points = 0

        rows.append({
            'Season': season,
            'Team': home_team,
            'MatchDate': date,
            'GoalsFor': row['FTHG'],
            'GoalsAgainst': row['FTAG'],
            'MatchPoints': home_points
        })

        away_team = row['AwayTeam']
        if home_result == 'A':
            away_points = 3
        elif home_result == 'D':
            away_points = 1
        else:
            away_points = 0

        rows.append({
            'Season': season,
            'Team': away_team,
            'MatchDate': date,
            'GoalsFor': row['FTAG'],
            'GoalsAgainst': row['FTHG'],
            'MatchPoints': away_points
        })

    team_matches = pd.DataFrame(rows)
    team_matches.sort_values(['Season','Team','MatchDate'], inplace=True)

    partial_list = []
    for (season, team), group in team_matches.groupby(['Season','Team']):
        first_N = group.iloc[:N]
        partial_points = first_N['MatchPoints'].sum()
        partial_gf     = first_N['GoalsFor'].sum()
        partial_ga     = first_N['GoalsAgainst'].sum()

        partial_list.append({
            'Season': season,
            'Team': team,
            'Partial_Matches': len(first_N),
            'Partial_Points': partial_points,
            'Partial_GoalsFor': partial_gf,
            'Partial_GoalsAgainst': partial_ga
        })

    partial_df = pd.DataFrame(partial_list)

    # Merge with full aggregator
    merged_partial = partial_df.merge(
        final_table[['Season','Team','Points']],
        on=['Season','Team'], how='left'
    )
    if 'Points' in merged_partial.columns:
        merged_partial.rename(columns={'Points':'Final_Points'}, inplace=True)

    print("Partial-season dataset columns:", merged_partial.columns.tolist())
    return merged_partial


# -----------------------------------------------------
# 3) Streamlit Data Loader
# -----------------------------------------------------
@st.cache_data
def load_new_data():
    players = pd.read_csv("Player Stats/Players.csv")
    squads = pd.read_csv("Team Stats/Squads.csv")
    all_matches_df = pd.read_csv("Team Stats/results.csv", encoding="latin-1")
    all_matches_df['DateTime'] = pd.to_datetime(all_matches_df['DateTime'], errors='coerce')

    final_table = build_final_table("Team Stats/results.csv")
    return players, squads, all_matches_df, final_table


# -----------------------------------------------------
# 4) Merge Squads with Full-Season Aggregation
# -----------------------------------------------------
def merge_squads_and_final_table(squads_df: pd.DataFrame, final_table_df: pd.DataFrame) -> pd.DataFrame:
    squads_df["Squad"] = squads_df["Squad"].astype(str).str.strip()
    final_table_df["Team"] = final_table_df["Team"].astype(str).str.strip()

    merged = squads_df.merge(
        final_table_df, left_on="Squad", right_on="Team", how="left"
    )

    # unify Points_x/y
    if 'Points_x' in merged.columns and 'Points_y' in merged.columns:
        merged['Points'] = merged['Points_x'].fillna(merged['Points_y'])
        merged.drop(['Points_x','Points_y'], axis=1, inplace=True)

    # unify GD_x/y if they exist
    if 'GD_x' in merged.columns and 'GD_y' in merged.columns:
        merged['GD'] = merged['GD_x'].fillna(merged['GD_y'])
        merged.drop(['GD_x','GD_y'], axis=1, inplace=True)

    print("Columns after merging (full-season):", merged.columns.tolist())
    return merged


# -----------------------------------------------------
# 5) Train Full-Season Model (Time-Based Split)
# -----------------------------------------------------
def train_full_season_model_time_split(df: pd.DataFrame):
    """
    We'll treat 'Points' as the TARGET only, not a feature.
    Features: GF, GA, GD, HS, HST, HC (no 'Points')
    We do a time-based split: train on <2019, test on >=2019
    We'll also print the shapes to see how big the sets are.
    """
    # 1) Ensure we have final points
    df = df.dropna(subset=['Points'])

    # 2) parse season to integer
    def parse_season_start(s):
        try:
            return int(s.split('-')[0])
        except:
            return np.nan

    df['SeasonStart'] = df['Season'].apply(parse_season_start)
    df.dropna(subset=['SeasonStart'], inplace=True)

    # Optional: remove very old seasons
    df = df[df['SeasonStart'] >= 2010]

    # 3) time-based split
    train_df = df[df['SeasonStart'] < 2021]
    test_df  = df[df['SeasonStart'] >= 2021]

    # 4) define features WITHOUT 'Points' or any final columns
    features = ["GF","GA","GD","HS","HST","HC"]
    # Check columns for debugging
    print("Available columns in merged_full:", df.columns.tolist())
    print("Using features:", features)

    X_train = train_df[features].fillna(0).astype(float)
    y_train = train_df['Points'].astype(float)

    X_test  = test_df[features].fillna(0).astype(float)
    y_test  = test_df['Points'].astype(float)

    print(f"train_df shape: {train_df.shape}, test_df shape: {test_df.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds_test = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)

    print(f"Time-based Full-season => MAE: {mae:.2f}, R2: {r2:.2f}")
    return model, scaler, mae, r2


# -----------------------------------------------------
# 6) Train Partial-Season Model
# -----------------------------------------------------
def train_partial_model(partial_df: pd.DataFrame):
    partial_df = partial_df.dropna(subset=['Final_Points'])
    features = ['Partial_Points','Partial_GoalsFor','Partial_GoalsAgainst']
    X = partial_df[features].fillna(0)
    y = partial_df['Final_Points']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Partial-season => MAE: {mae:.2f}, R2: {r2:.2f}")
    return model, scaler, mae, r2


# -----------------------------------------------------
# 7) Predict full-season aggregator
# -----------------------------------------------------
def predict_full_season(model, scaler, df: pd.DataFrame):
    features = ["GF","GA","GD","HS","HST","HC"]
    X = df[features].fillna(0).astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    df["PredictedPoints"] = preds
    df.sort_values(by="PredictedPoints", ascending=False, inplace=True)
    return df


# -----------------------------------------------------
# 8) Predict partial-season -> final outcome
# -----------------------------------------------------
def predict_partial_season(model, scaler, partial_df: pd.DataFrame):
    features = ['Partial_Points','Partial_GoalsFor','Partial_GoalsAgainst']
    X = partial_df[features].fillna(0).astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    partial_df["PredictedFinalPoints"] = preds
    partial_df.sort_values(by="PredictedFinalPoints", ascending=False, inplace=True)
    return partial_df


# -----------------------------------------------------
# 9) Helper code for next 10 years winners
# -----------------------------------------------------
def compute_team_averages(full_df):
    # We'll take average of GF, GA, GD, HS, HST, HC
    group = full_df.groupby("Team", as_index=False).agg({
        "GF": "mean",
        "GA": "mean",
        "GD": "mean",
        "HS": "mean",
        "HST": "mean",
        "HC": "mean"
    })
    group.rename(columns={
        "GF": "avg_GF",
        "GA": "avg_GA",
        "GD": "avg_GD",
        "HS": "avg_HS",
        "HST": "avg_HST",
        "HC": "avg_HC"
    }, inplace=True)
    return group


def build_future_scenarios(team_averages, start_year=2024, end_year=2033):
    """
    We'll add random variation ±10% around each average stat,
    so each future year is not identical for the same team.
    """
    rows = []
    rng = np.random.default_rng(seed=42)  # for reproducibility
    for year in range(start_year, end_year+1):
        season_str = f"{year}-{(year+1)%100:02d}"
        for idx, row in team_averages.iterrows():
            # Add random variation ±10%
            gf = row["avg_GF"] * rng.uniform(0.9, 1.1)
            ga = row["avg_GA"] * rng.uniform(0.9, 1.1)
            gd = row["avg_GD"] * rng.uniform(0.9, 1.1)
            hs = row["avg_HS"] * rng.uniform(0.9, 1.1)
            hst= row["avg_HST"]* rng.uniform(0.9, 1.1)
            hc = row["avg_HC"] * rng.uniform(0.9, 1.1)

            rows.append({
                "Season": season_str,
                "Team": row["Team"],
                "GF": gf,
                "GA": ga,
                "GD": gd,
                "HS": hs,
                "HST": hst,
                "HC": hc
            })
    return pd.DataFrame(rows)


def predict_future_winners(model, scaler, future_df):
    all_results = []
    for season in sorted(future_df["Season"].unique()):
        chunk = future_df[future_df["Season"] == season].copy()
        features = ["GF","GA","GD","HS","HST","HC"]
        X = chunk[features].fillna(0).astype(float)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        chunk["PredictedPoints"] = preds
        chunk.sort_values(by="PredictedPoints", ascending=False, inplace=True)
        champ = chunk.iloc[0]["Team"]
        champ_pts = chunk.iloc[0]["PredictedPoints"]
        all_results.append((season, champ, champ_pts))
    return all_results


# -----------------------------------------------------
# 10) Streamlit App
# -----------------------------------------------------
def main():
    st.title("Premier League Predictor")

    # 1) Load data
    players, squads, all_matches_df, final_table = load_new_data()
    merged_full = merge_squads_and_final_table(squads, final_table)

    # 2) Train Full-Season Model
    st.subheader("Train Full-Season Model (Time-Based Split)")
    fs_model, fs_scaler, fs_mae, fs_r2 = train_full_season_model_time_split(merged_full)
    st.write(f"Time-based Full-season => MAE: {fs_mae:.2f}, R2: {fs_r2:.2f}")

    # 3) User options
    user_option = st.selectbox("Select an option:", [
        "Predict full-season results",
        "Predict partial-season final outcome (N=24)",
        "Predict next 10 years winners"
    ])

    if user_option == "Predict full-season results":
        st.write("Predicting final points using GF, GA, GD, HS, HST, HC (no leakage).")
        results_full = predict_full_season(fs_model, fs_scaler, merged_full.copy())
        st.dataframe(results_full[["Season","Team","Points","GF","GA","GD","HS","HST","HC","PredictedPoints"]])
        top_team = results_full.iloc[0]["Team"]
        top_pts = results_full.iloc[0]["PredictedPoints"]
        st.write(f"**Predicted Champion**: {top_team} with ~{top_pts:.1f} points.")

    elif user_option == "Predict partial-season final outcome (N=24)":
        st.write("Building partial dataset for first 24 matches, then predicting final points.")
        partial_24_df = build_partial_dataset(all_matches_df, final_table, N=24)
        ps_model, ps_scaler, ps_mae, ps_r2 = train_partial_model(partial_24_df)
        st.write(f"Partial-season => MAE: {ps_mae:.2f}, R2: {ps_r2:.2f}")

        partial_preds = predict_partial_season(ps_model, ps_scaler, partial_24_df.copy())
        st.dataframe(partial_preds[[
            "Season","Team","Partial_Matches","Partial_Points","Final_Points","PredictedFinalPoints"
        ]])
        if not partial_preds.empty:
            best_team = partial_preds.iloc[0]["Team"]
            best_pts  = partial_preds.iloc[0]["PredictedFinalPoints"]
            st.write(f"**Partial-Season Predicted Champion**: {best_team} with ~{best_pts:.1f} final pts")

    elif user_option == "Predict next 10 years winners":
        st.write("Predicting future seasons (2024–2033) with random variation on each team's average stats.")
        team_avgs = compute_team_averages(merged_full)
        future_df = build_future_scenarios(team_avgs, 2024, 2033)
        future_results = predict_future_winners(fs_model, fs_scaler, future_df)

        for (season, champ, pts) in future_results:
            st.write(f"{season} Winner: **{champ}** with approx. {pts:.1f} predicted points")


if __name__ == "__main__":
    main()
