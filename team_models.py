import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PSYCH_FEATURE_COLS = [
    "HomeForm", "AwayForm",
    "HomeFouls", "AwayFouls",
    "HomeYellows", "AwayYellows",
    "HomePressure", "AwayPressure",
    "FormDiff", "FoulsDiff", "YellowsDiff", "PressureDiff"
]
# === Utility ===
def read_s3_csv(bucket: str, key: str) -> pd.DataFrame:
    """
    function used to read a CSV file from AWS S3 using boto3
    Args:
        - bucket: name of the S3 bucket
        - key: path to the CSV file in the bucket
    Returns:
        - pd.DataFrame: parsed DataFrame from the CSV
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response['Body'].read()), encoding="latin-1")

def compute_recent_form(df, n=5):
    """
    function used to calculate recent form (points per game) for each team over last n matches
    Args:
        - df: full match-level dataset including home/away teams and points
        - n: number of games to include in the rolling average
    Returns:
        - pd.DataFrame: contains Season, Team, and RecentFormPPG for latest match
    """
    df = df.sort_values(["Season", "Date"])
    all_matches = []

    for team_col, is_home in [("HomeTeam", True), ("AwayTeam", False)]:
        for team in df[team_col].unique():
            team_df = df[df[team_col] == team].copy()
            team_df["Team"] = team
            team_df["IsHome"] = is_home
            team_df["Points"] = team_df["HomeTeamPoints"] if is_home else team_df["AwayTeamPoints"]
            team_df = team_df[["Season", "Date", "MatchWeek", "Team", "Points"]]
            all_matches.append(team_df)

    all_df = pd.concat(all_matches)
    all_df = all_df.sort_values(["Season", "Team", "Date"])
    all_df["RecentFormPPG"] = all_df.groupby(["Season", "Team"])["Points"].transform(lambda x: x.rolling(n, min_periods=1).mean())
    
    # Take the most recent matchweek value for each team
    return all_df.groupby(["Season", "Team"]).tail(1)[["Season", "Team", "RecentFormPPG"]]

def compute_rolling_goal_stats(df, n=5):
    """
    function used to compute rolling goal statistics (scored and conceded) for each team
    Args:
        - df: full match-level dataset with goals and date
        - n: window size for rolling average
    Returns:
        - pd.DataFrame: contains average goals scored and conceded per team per season
    """
    df = df.sort_values(["Season", "Date"])
    goal_rows = []

    for team_col, is_home in [("HomeTeam", True), ("AwayTeam", False)]:
        for team in df[team_col].unique():
            team_df = df[df[team_col] == team].copy()
            team_df["Team"] = team
            team_df["GoalsScored"] = df["FTHG"] if is_home else df["FTAG"]
            team_df["GoalsConceded"] = df["FTAG"] if is_home else df["FTHG"]
            team_df = team_df[["Season", "Date", "Team", "GoalsScored", "GoalsConceded"]]
            goal_rows.append(team_df)

    all_goals = pd.concat(goal_rows).sort_values(["Season", "Team", "Date"])
    all_goals["AvgGoalsScored"] = all_goals.groupby(["Season", "Team"])["GoalsScored"].transform(lambda x: x.rolling(n, min_periods=1).mean())
    all_goals["AvgGoalsConceded"] = all_goals.groupby(["Season", "Team"])["GoalsConceded"].transform(lambda x: x.rolling(n, min_periods=1).mean())

    return all_goals.groupby(["Season", "Team"]).tail(1)[["Season", "Team", "AvgGoalsScored", "AvgGoalsConceded"]]

def build_goal_stats_df(matches_df, latest_season):
    """
    function used to build goal statistics per team for the latest season
    Args:
        - matches_df: full match-level dataset
        - latest_season: string representing most recent season
    Returns:
        - pd.DataFrame: contains team-level goal stats and form for latest season
    """
    # Compute rolling averages of goals scored and conceded
    rolling = compute_rolling_goal_stats(matches_df)
    recent_form = compute_recent_form(matches_df)

    # Merge rolling xG and recent form
    goal_stats = rolling.merge(recent_form, on=["Season", "Team"], how="left")

    # Filter to latest season only
    goal_stats = goal_stats[goal_stats["Season"] == latest_season]

    return goal_stats.reset_index(drop=True)


def standardize_columns(df):
    """
    function used to standardize match data column names across different sources
    Args:
        - df: raw DataFrame from CSV
    Returns:
        - pd.DataFrame: DataFrame with renamed standard columns
    """
    rename_map = {
        "FullTimeHomeTeamGoals": "FTHG",
        "FullTimeAwayTeamGoals": "FTAG",
        "FullTimeResult": "FTR",
        "HomeTeamShots": "HS",
        "AwayTeamShots": "AS",
        "HomeTeamShotsOnTarget": "HST",
        "AwayTeamShotsOnTarget": "AST",
        "HomeTeamCorners": "HC",
        "AwayTeamCorners": "AC",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

def normalize_season(season_str):
    """
    function used to normalize season strings to consistent 'YYYY-YYYY' format
    Args:
        - season_str: raw season string (e.g., '2024-25')
    Returns:
        - str: normalized full season string (e.g., '2024-2025')
    """
    season_str = str(season_str).strip()
    if "-" in season_str and len(season_str.split("-")[1]) == 2:
        # Convert "2024-25" to "2024-2025"
        start, end = season_str.split("-")
        end = str(int(start[:2] + end))
        return f"{start}-{end}"
    return season_str

# === Load datasets ===
@st.cache_data
def load_data_sources():
    """
    function used to load and clean all team-level data from S3 into memory
    Args:
        None
    Returns:
        - combined: full combined dataset of matches + results
        - past: past season summary data
        - remaining: fixture data for remaining games
        - latest_season: latest detected season string
    """
    matches = read_s3_csv("finalyearproject2025", "teamData/Results/PremierLeague.csv")
    results = read_s3_csv("finalyearproject2025", "teamData/Results/results.csv")
    past = read_s3_csv("finalyearproject2025", "teamData/Results/PastResults.csv")
    remaining = read_s3_csv("finalyearproject2025", "teamData/Results/RemainingMatches.csv")

    matches = standardize_columns(matches)
    results = standardize_columns(results)

    matches["Season"] = matches["Season"].apply(normalize_season)
    results["Season"] = results["Season"].apply(normalize_season)
    past["Season"] = past["Season"].apply(normalize_season)
    remaining["Season"] = remaining["Season"].apply(normalize_season)

    combined = pd.concat([matches, results], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined["MatchWeek"] = combined.groupby("Season")["Date"].rank(method="dense").astype(int)

    latest_season = sorted(combined["Season"].dropna().unique())[-1]
    st.write("🔍 Latest season detected:", latest_season)

    return combined, past, remaining, latest_season

# === Add points from FTR column ===
def add_match_points(df):
    """
    function used to add home and away team points based on full-time result
    Args:
        - df: match-level DataFrame with result column
    Returns:
        - pd.DataFrame: same DataFrame with HomeTeamPoints and AwayTeamPoints columns
    """
    if "FTR" not in df.columns and "FullTimeResult" in df.columns:
        df["FTR"] = df["FullTimeResult"]

    def home_pts(ftr): return 3 if ftr == "H" else 1 if ftr == "D" else 0
    def away_pts(ftr): return 3 if ftr == "A" else 1 if ftr == "D" else 0

    if "FTR" in df.columns:
        df["HomeTeamPoints"] = df["FTR"].apply(home_pts)
        df["AwayTeamPoints"] = df["FTR"].apply(away_pts)
    return df

# === Model training ===
def train_model_on_remaining_points(df, past_df, remaining, season, cutoff_week=34):
    """
    function used to train ML model to estimate remaining points per team for the season
    Args:
        - df: historical match data
        - past_df: past season final points
        - remaining: fixture list
        - season: current season being predicted
        - cutoff_week: maximum matchweek to consider from historical data
    Returns:
        - model: trained XGBoostRegressor model
        - scaler: standard scaler used for feature normalization
    """
    # Use only past seasons
    df = df[df["Season"] != season]
    df = df[df["MatchWeek"] <= cutoff_week]
    df = add_match_points(df)

    required_cols = {"FTHG", "FTAG", "HS", "HST", "HC", "AS", "AST", "AC"}
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # === Aggregate stats ===
    home = df.groupby(["Season", "HomeTeam"]).agg({
        "HomeTeamPoints": "sum", "FTHG": "sum", "FTAG": "sum",
        "HS": "sum", "HST": "sum", "HC": "sum"
    }).reset_index().rename(columns={"HomeTeam": "Team"})

    away = df.groupby(["Season", "AwayTeam"]).agg({
        "AwayTeamPoints": "sum", "FTAG": "sum", "FTHG": "sum",
        "AS": "sum", "AST": "sum", "AC": "sum"
    }).reset_index().rename(columns={"AwayTeam": "Team"})

    combined = pd.concat([home, away])
    agg = combined.groupby(["Season", "Team"], as_index=False).sum()
    agg["GD"] = agg["FTHG"] - agg["FTAG"]
    agg["PartialPoints"] = agg["HomeTeamPoints"] + agg["AwayTeamPoints"]

    # === Estimate MatchesLeft as 38 - games played
    agg["GamesPlayed"] = agg["HomeTeamPoints"].notna().astype(int) + agg["AwayTeamPoints"].notna().astype(int)
    agg["MatchesLeft"] = 38 - agg["GamesPlayed"]

    # === Merge with Past Final Points
    past_df = past_df.rename(columns={"SquadName": "Team", "Pts": "FinalPoints"})
    past_df["Season"] = past_df["Season"].apply(normalize_season)
    merged = agg.merge(past_df[["Season", "Team", "FinalPoints"]], on=["Season", "Team"], how="inner")

    merged["RemainingPoints"] = merged["FinalPoints"] - merged["PartialPoints"]
    merged["RemainingPPG"] = merged["RemainingPoints"] / merged["MatchesLeft"]

    # === Add recent form
    recent_form = compute_recent_form(df)
    merged = merged.merge(recent_form, on=["Season", "Team"], how="left").fillna(0)

    # === Features & target
    features = ["PartialPoints", "FTHG", "FTAG", "GD", "HS", "HST", "HC", "MatchesLeft", "RecentFormPPG"]
    X = merged[features].fillna(0)
    y = merged["RemainingPPG"]

    valid_idx = y.notna() & ~y.isin([np.inf, -np.inf])
    X = X.loc[valid_idx].replace([np.inf, -np.inf], 0)
    y = y.loc[valid_idx]

    if X.empty or y.empty:
        st.error("⚠️ No data available to train model.")
        return None, None

    scaler = StandardScaler()
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    model.fit(scaler.fit_transform(X), y)

    preds = model.predict(scaler.transform(X)) * merged.loc[valid_idx, "MatchesLeft"]
    mae = mean_absolute_error(merged.loc[valid_idx, "RemainingPoints"], preds)
    r2 = r2_score(merged.loc[valid_idx, "RemainingPoints"], preds)

    st.caption(f"📈 MAE: {mae:.2f} | R²: {r2:.2f}")
    return model, scaler

# === Predict for latest season ===
def build_partial_features(df, season, remaining, cutoff_week=34):
    """
    function used to build team-level stats for current season up to cutoff week
    Args:
        - df: match data
        - season: current season
        - remaining: fixture list
        - cutoff_week: maximum matchweek to include
    Returns:
        - pd.DataFrame: team-level partial season features
    """

    df = df[df["Season"] == season]
    df = add_match_points(df)

    required_cols = {"FTHG", "FTAG", "HS", "HST", "HC", "AS", "AST", "AC"}
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Home aggregation
    home = df.groupby("HomeTeam").agg({
        "HomeTeamPoints": "sum", "FTHG": "sum", "FTAG": "sum",
        "HS": "sum", "HST": "sum", "HC": "sum"
    }).reset_index().rename(columns={"HomeTeam": "Team"})

    # Away aggregation
    away = df.groupby("AwayTeam").agg({
        "AwayTeamPoints": "sum", "FTAG": "sum", "FTHG": "sum",
        "AS": "sum", "AST": "sum", "AC": "sum"
    }).reset_index().rename(columns={"AwayTeam": "Team"})

    # Merge and aggregate
    combined = pd.concat([home, away])
    agg = combined.groupby("Team", as_index=False).sum()
    agg["GD"] = agg["FTHG"] - agg["FTAG"]
    agg["PartialPoints"] = agg["HomeTeamPoints"] + agg["AwayTeamPoints"]

    remaining_counts = remaining[remaining["Season"] == season].copy()
    remaining_counts = pd.concat([
        remaining_counts["HomeTeam"],
        remaining_counts["AwayTeam"]
    ]).value_counts().reset_index()
    remaining_counts.columns = ["Team", "MatchesLeft"]
    agg = agg.merge(remaining_counts, on="Team", how="left").fillna(0)
    agg["MatchesLeft"] = agg["MatchesLeft"].astype(int)


    # Add recent form
    recent_form = compute_recent_form(df)
    recent_form = recent_form[recent_form["Season"] == season]
    agg = agg.merge(recent_form[["Team", "RecentFormPPG"]], on="Team", how="left").fillna(0)

    return agg


# === Final prediction ===
def predict_final(model, scaler, df):
    """
    function used to generate final league predictions using trained model and partial features
    Args:
        - model: trained model
        - scaler: standard scaler for features
        - df: partial season feature set
    Returns:
        None
    """
    # 1) Prepare feature matrix
    features = ["PartialPoints", "FTHG", "FTAG", "GD", "HS", "HST", "HC", "MatchesLeft", "RecentFormPPG"]
    X = df[features].fillna(0)

    if X.empty:
        st.error("⚠️ No data to predict.")
        return

    # 2) Model prediction: average remaining PPG × remaining games
    raw_remaining = model.predict(scaler.transform(X)) * df["MatchesLeft"]

    # 3) Clip to feasible range [0, 3*MatchesLeft]
    max_remaining = df["MatchesLeft"] * 3
    clipped = np.minimum(np.maximum(raw_remaining, 0), max_remaining)

    # 4) Round up to whole points
    df["PredictedRemaining"] = np.ceil(clipped).astype(int)

    # 5) Final and max points
    df["PredictedFinalPoints"] = df["PartialPoints"] + df["PredictedRemaining"]
    df["MaxFinalPoints"] = df["PartialPoints"] + max_remaining

    # 6) Sort and reset index, add Rank
    df = df.sort_values("PredictedFinalPoints", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    # 7) Display nicely
    st.dataframe(df[[
    "Rank", "Team", "PartialPoints", "MatchesLeft", "RecentFormPPG",
    "PredictedRemaining", "PredictedFinalPoints", "MaxFinalPoints"
    ]], use_container_width=True, hide_index=True)

    #st.write("🧪 Raw prediction input", df[["Team", "PartialPoints", "MatchesLeft", "RecentFormPPG"]])


    # 8) Champion call‐out
    champ = df.iloc[0]
    st.success(f"🏆 Predicted Champion: **{champ['Team']}** with **{int(champ['PredictedFinalPoints'])}** points")

def display_psychological_table(df, season, remaining_df, return_df=False):
    """
    function used to compute per-team psychological and discipline features including:
    fouls, cards, pressure index, referee strictness, and tension index
    Args:
        - df: full match data
        - season: current season string
        - remaining_df: remaining fixtures
        - return_df: whether to return the table or display via Streamlit
    Returns:
        - Optional[pd.DataFrame]: if return_df is True, returns the computed DataFrame
    """
    df = df[df["Season"] == season].copy()
    stat_cols = [
        "HomeTeamFouls", "AwayTeamFouls",
        "HomeTeamYellowCards", "AwayTeamYellowCards",
        "HomeTeamRedCards", "AwayTeamRedCards"
    ]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0

    # --- Per-team averages ---
    def agg_stats(group_col, prefix):
        return df.groupby(group_col).agg({
            f"{prefix}Fouls": "mean",
            f"{prefix}YellowCards": "mean",
            f"{prefix}RedCards": "mean"
        }).rename(columns={
            f"{prefix}Fouls": "FoulsPerGame",
            f"{prefix}YellowCards": "YellowCardsPerGame",
            f"{prefix}RedCards": "RedCardsPerGame"
        })

    home_stats = agg_stats("HomeTeam", "HomeTeam")
    away_stats = agg_stats("AwayTeam", "AwayTeam")
    team_stats = (home_stats.add(away_stats, fill_value=0) / 2).reset_index()
    team_stats = team_stats.rename(columns={"index": "Team", "HomeTeam": "Team", "AwayTeam": "Team"})

    # --- Discipline Index ---
    team_stats["DisciplineIndex"] = (
        1.0 * team_stats["FoulsPerGame"] +
        2.0 * team_stats["YellowCardsPerGame"] +
        3.0 * team_stats["RedCardsPerGame"]
    )

    # --- Recent form ---
    recent_form = compute_recent_form(df).copy()
    recent_form = recent_form[recent_form["Season"] == season][["Team", "RecentFormPPG"]]
    team_stats = team_stats.merge(recent_form, on="Team", how="left").fillna(0)

    # --- Pressure flags ---
    all_points = pd.concat([
        df[["HomeTeam", "HomeTeamPoints"]].rename(columns={"HomeTeam": "Team", "HomeTeamPoints": "Points"}),
        df[["AwayTeam", "AwayTeamPoints"]].rename(columns={"AwayTeam": "Team", "AwayTeamPoints": "Points"})
    ])
    team_points = all_points.groupby("Team")["Points"].sum().reset_index()
    team_points["Rank"] = team_points["Points"].rank(ascending=False, method="min")
    team_points["HighPressureFlag"] = team_points["Rank"].apply(
        lambda r: 1 if r <= 5 or r > len(team_points) - 5 else 0
    )
    team_stats = team_stats.merge(team_points[["Team", "HighPressureFlag"]], on="Team", how="left")

    # --- Opponent Aggression & Tension Index ---
    expected_opp_cols = ["Team", "Opponent1", "Opponent2", "Opponent3", "Opponent4"]
    if all(col in remaining_df.columns for col in expected_opp_cols):
        opp_stats = team_stats.set_index("Team")[["FoulsPerGame", "YellowCardsPerGame", "RedCardsPerGame"]]

        def compute_opponent_score(row):
            opps = [row.get(f"Opponent{i}") for i in range(1, 5) if pd.notnull(row.get(f"Opponent{i}"))]
            if not opps:
                return pd.Series([0, 0, 0], index=["OpponentFoulsPG", "OpponentYellowsPG", "OpponentRedsPG"])
            scores = opp_stats.loc[opps].mean()
            return pd.Series({
                "OpponentFoulsPG": scores["FoulsPerGame"],
                "OpponentYellowsPG": scores["YellowCardsPerGame"],
                "OpponentRedsPG": scores["RedCardsPerGame"]
            })

        opp_score_df = remaining_df[remaining_df["Season"] == season][expected_opp_cols].copy()
        opp_scores = opp_score_df.set_index("Team").apply(compute_opponent_score, axis=1).reset_index()
        team_stats = team_stats.merge(opp_scores, on="Team", how="left")

        # Opponent pressure + tension index
        pressure_map = team_points.set_index("Team")["HighPressureFlag"].to_dict()

        def avg_opp_pressure(row):
            opps = [row.get(f"Opponent{i}") for i in range(1, 5) if pd.notnull(row.get(f"Opponent{i}"))]
            return np.mean([pressure_map.get(opp, 0) for opp in opps]) if opps else 0

        opp_score_df["OpponentPressure"] = opp_score_df.apply(avg_opp_pressure, axis=1)
        team_stats = team_stats.merge(opp_score_df[["Team", "OpponentPressure"]], on="Team", how="left")
        team_stats["TensionIndex"] = team_stats["HighPressureFlag"] + team_stats["OpponentPressure"]
    else:
        team_stats["OpponentFoulsPG"] = 0
        team_stats["OpponentYellowsPG"] = 0
        team_stats["OpponentRedsPG"] = 0
        team_stats["OpponentPressure"] = 0
        team_stats["TensionIndex"] = team_stats["HighPressureFlag"]

    # --- Referee Strictness ---
    referee_cards = df.groupby("Referee").agg({
        "HomeTeamYellowCards": "mean",
        "AwayTeamYellowCards": "mean",
        "HomeTeamRedCards": "mean",
        "AwayTeamRedCards": "mean"
    }).mean(axis=1).reset_index(name="RefereeStrictness")

    df_ref = df[["Referee", "HomeTeam", "AwayTeam"]].dropna()
    home_ref = df_ref[["Referee", "HomeTeam"]].rename(columns={"HomeTeam": "Team"})
    away_ref = df_ref[["Referee", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})
    team_ref = pd.concat([home_ref, away_ref])
    team_strictness = team_ref.merge(referee_cards, on="Referee", how="left").groupby("Team")["RefereeStrictness"].mean().reset_index()
    team_stats = team_stats.merge(team_strictness, on="Team", how="left")

    # --- Display or return ---
    if return_df:
        return team_stats

    st.dataframe(
        team_stats.sort_values("TensionIndex", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

def train_discipline_risk_model(df, season, cutoff_week=34):
    """
    function used to train two separate ML regressors for predicting remaining yellow and red cards
    Args:
        - df: match dataset
        - season: current season
        - cutoff_week: only use data before this matchweek
    Returns:
        - model_yellow: trained regressor for yellow cards
        - model_red: trained regressor for red cards
        - scaler: fitted scaler for input features
    """
    df = df[df["Season"] != season]
    df = df[df["MatchWeek"] <= cutoff_week]

    stat_cols = [
        "HomeTeamFouls", "AwayTeamFouls",
        "HomeTeamYellowCards", "AwayTeamYellowCards",
        "HomeTeamRedCards", "AwayTeamRedCards"
    ]
    for col in stat_cols:
        df[col] = df.get(col, 0)
    df = add_match_points(df)
    recent_form = compute_recent_form(df)

    # Aggregate fouls/cards per team
    home = df.groupby("HomeTeam").agg({
        "HomeTeamFouls": "mean",
        "HomeTeamYellowCards": "mean",
        "HomeTeamRedCards": "mean"
    }).rename(columns=lambda x: x.replace("HomeTeam", "")).reset_index().rename(columns={"HomeTeam": "Team"})

    away = df.groupby("AwayTeam").agg({
        "AwayTeamFouls": "mean",
        "AwayTeamYellowCards": "mean",
        "AwayTeamRedCards": "mean"
    }).rename(columns=lambda x: x.replace("AwayTeam", "")).reset_index().rename(columns={"AwayTeam": "Team"})

    team_stats = pd.concat([home, away]).groupby("Team", as_index=False).mean()

    # Merge with recent form
    recent_form = recent_form[recent_form["Season"] != season]
    team_stats = team_stats.merge(recent_form[["Team", "RecentFormPPG"]], on="Team", how="left")

    # Add pressure flag
    all_points = pd.concat([
        df[["HomeTeam", "HomeTeamPoints"]].rename(columns={"HomeTeam": "Team", "HomeTeamPoints": "Points"}),
        df[["AwayTeam", "AwayTeamPoints"]].rename(columns={"AwayTeam": "Team", "AwayTeamPoints": "Points"})
    ])
    team_points = all_points.groupby("Team")["Points"].sum().reset_index()
    team_points["Rank"] = team_points["Points"].rank(ascending=False, method="min")
    team_points["HighPressureFlag"] = team_points["Rank"].apply(
        lambda r: 1 if r <= 5 or r > len(team_points) - 5 else 0
    )
    team_stats = team_stats.merge(team_points[["Team", "HighPressureFlag"]], on="Team", how="left")
    # Add referee strictness
    referee_cards = df.groupby("Referee").agg({
        "HomeTeamYellowCards": "mean",
        "AwayTeamYellowCards": "mean",
        "HomeTeamRedCards": "mean",
        "AwayTeamRedCards": "mean"
    }).mean(axis=1).reset_index(name="RefereeStrictness")

    df_ref = df[["Referee", "HomeTeam", "AwayTeam"]].dropna()
    home_ref = df_ref[["Referee", "HomeTeam"]].rename(columns={"HomeTeam": "Team"})
    away_ref = df_ref[["Referee", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})
    team_ref = pd.concat([home_ref, away_ref])
    team_strictness = team_ref.merge(referee_cards, on="Referee", how="left").groupby("Team")["RefereeStrictness"].mean().reset_index()
    team_stats = team_stats.merge(team_strictness, on="Team", how="left")

    # Labels
    team_stats["RemainingYellowCards"] = team_stats["YellowCards"] * (38 - cutoff_week)
    team_stats["RemainingRedCards"] = team_stats["RedCards"] * (38 - cutoff_week)

    # Drop NaNs and Inf
    team_stats = team_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "Fouls", "YellowCards", "RedCards", "RecentFormPPG",
        "RemainingYellowCards", "RemainingRedCards"
    ])

    # Features
    features = ["Fouls", "YellowCards", "RedCards", "RecentFormPPG", "HighPressureFlag", "RefereeStrictness"]
    X = team_stats[features]
    y_yellow = team_stats["RemainingYellowCards"]
    y_red = team_stats["RemainingRedCards"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Grid Search Parameters ===
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5]
    }

    # GridSearch for Yellow Cards
    grid_yellow = GridSearchCV(
        estimator=XGBRegressor(),
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )
    grid_yellow.fit(X_scaled, y_yellow)
    model_yellow = grid_yellow.best_estimator_

    # GridSearch for Red Cards
    grid_red = GridSearchCV(
        estimator=XGBRegressor(),
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )
    grid_red.fit(X_scaled, y_red)
    model_red = grid_red.best_estimator_

    st.caption(f"🟨 Best YellowCard Model MAE: {-grid_yellow.best_score_:.2f}")
    st.caption(f"🟥 Best RedCard Model MAE: {-grid_red.best_score_:.2f}")
    return model_yellow, model_red, scaler


def get_team_stats_lookup(matches, season, remaining):
    """
    function used to fetch team-level stats with renamed and indexed discipline and form metrics
    Args:
        - matches: full match data
        - season: current season string
        - remaining: remaining fixture list
    Returns:
        - pd.DataFrame: indexed by team with pressure, discipline, and form stats
    """
    stats = display_psychological_table(matches, season, remaining, return_df=True).copy()
    stats = stats.rename(columns={
        "FoulsPerGame": "Fouls",
        "YellowCardsPerGame": "YellowCards",
        "RedCardsPerGame": "RedCards",
        "HighPressureFlag": "Pressure"
    })
    return stats.set_index("Team")


def train_psych_match_predictor(df, season, cutoff_week=34):
    """
    function used to train a classifier to predict match outcomes using psychological and performance features
    Args:
        - df: historical match data
        - season: current season to exclude from training
        - cutoff_week: matchweek cutoff for training data
    Returns:
        - model: trained XGBClassifier for match outcome prediction
    """
    df = df[df["Season"] != season]
    df = df[df["MatchWeek"] <= cutoff_week]
    df = add_match_points(df)

    required_cols = [
        "HomeTeamFouls", "AwayTeamFouls", "HomeTeamYellowCards", "AwayTeamYellowCards",
        "FullTimeHomeTeamGoals", "FullTimeAwayTeamGoals",
        "HalfTimeHomeTeamGoals", "HalfTimeAwayTeamGoals",
        "HomeTeamCorners", "AwayTeamCorners"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    recent_form = compute_recent_form(df)
    team_form = recent_form.set_index(["Season", "Team"])["RecentFormPPG"]

    def get_form(team, season):
        return team_form.get((season, team), 1.3)

    def get_features(row):
        home_form = get_form(row["HomeTeam"], row["Season"])
        away_form = get_form(row["AwayTeam"], row["Season"])

        return pd.Series({
            "HomeForm": home_form,
            "AwayForm": away_form,
            "HomeFouls": row["HomeTeamFouls"],
            "AwayFouls": row["AwayTeamFouls"],
            "HomeYellows": row["HomeTeamYellowCards"],
            "AwayYellows": row["AwayTeamYellowCards"],
            "HomePressure": 1 if row["HomeTeamPoints"] > 60 else 0,
            "AwayPressure": 1 if row["AwayTeamPoints"] > 60 else 0,
            "FormDiff": home_form - away_form,
            "FoulsDiff": row["HomeTeamFouls"] - row["AwayTeamFouls"],
            "YellowsDiff": row["HomeTeamYellowCards"] - row["AwayTeamYellowCards"],
            "PressureDiff": (1 if row["HomeTeamPoints"] > 60 else 0) - (1 if row["AwayTeamPoints"] > 60 else 0),
            "HomeGoals": row["FullTimeHomeTeamGoals"],
            "AwayGoals": row["FullTimeAwayTeamGoals"],
            "GoalDiff": row["FullTimeHomeTeamGoals"] - row["FullTimeAwayTeamGoals"],
            "HTGoalDiff": row["HalfTimeHomeTeamGoals"] - row["HalfTimeAwayTeamGoals"],
            "CornerDiff": row["HomeTeamCorners"] - row["AwayTeamCorners"]
        })

    features_df = df.apply(get_features, axis=1, result_type='expand')
    labels = df["FTR"].map({"H": 0, "D": 1, "A": 2})

    class_weights = labels.value_counts(normalize=True).to_dict()
    sample_weights = labels.map(lambda x: 1.0 / class_weights.get(x, 1.0))

    features_to_use = features_df.columns.tolist()
    model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric="mlogloss")
    model.fit(features_df[features_to_use], labels, sample_weight=sample_weights)

    preds_proba = model.predict_proba(features_df[features_to_use])
    preds = model.predict(features_df[features_to_use])
    acc = accuracy_score(labels, preds)
    loss = log_loss(labels, preds_proba)

    st.caption(f"📊 Psychological Match Model Accuracy: {acc*100:.1f}%, Log Loss: {loss:.4f}")
    importances = pd.Series(model.feature_importances_, index=features_to_use).sort_values(ascending=False)
    # st.bar_chart(importances)

    return model

def predict_remaining_matches_with_psych(matches, remaining_df, model, season):
    """
    function used to predict outcomes of remaining matches using psychological model
    Args:
        - matches: full match-level dataset
        - remaining_df: fixture list for the current season
        - model: trained psychological match predictor
        - season: current season string
    Returns:
        - pd.DataFrame: remaining fixtures with predicted win/draw probabilities
    """
    df = remaining_df[remaining_df["Season"] == season].copy()
    recent_form = compute_recent_form(matches)
    team_form = recent_form.set_index(["Season", "Team"])["RecentFormPPG"]

    def get_form(team, season):
        return team_form.get((season, team), 1.3)

    def get_features(row):
        home_form = get_form(row["HomeTeam"], row["Season"])
        away_form = get_form(row["AwayTeam"], row["Season"])
        home_fouls = 10
        away_fouls = 10
        home_yellows = 1.5
        away_yellows = 1.5
        home_goals = 1.6
        away_goals = 1.1
        home_ht_goals = 0.8
        away_ht_goals = 0.6
        home_corners = 5.5
        away_corners = 4.2

        return pd.Series({
            "HomeForm": home_form,
            "AwayForm": away_form,
            "HomeFouls": home_fouls,
            "AwayFouls": away_fouls,
            "HomeYellows": home_yellows,
            "AwayYellows": away_yellows,
            "HomePressure": 1 if home_form > 1.6 else 0,
            "AwayPressure": 1 if away_form > 1.6 else 0,
            "FormDiff": home_form - away_form,
            "FoulsDiff": home_fouls - away_fouls,
            "YellowsDiff": home_yellows - away_yellows,
            "PressureDiff": (1 if home_form > 1.6 else 0) - (1 if away_form > 1.6 else 0),
            "HomeGoals": home_goals,
            "AwayGoals": away_goals,
            "GoalDiff": home_goals - away_goals,
            "HTGoalDiff": home_ht_goals - away_ht_goals,
            "CornerDiff": home_corners - away_corners
        })

    features_df = df.apply(get_features, axis=1, result_type='expand')
    preds = model.predict_proba(features_df.astype(float))

    output_df = df[["Date", "HomeTeam", "AwayTeam"]].copy()
    output_df[["HomeWinProb", "DrawProb", "AwayWinProb"]] = np.round(preds * 100, 1)
    return output_df.sort_values("HomeWinProb", ascending=False)
    

def plot_discipline_predictions(disc_df):
    """
    function used to plot predicted yellow and red cards for each team
    Args:
        - disc_df: discipline prediction DataFrame including team and card columns
    Returns:
        None
"""
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = disc_df.sort_values("PredictedYellowCards", ascending=False)
    ax.bar(sorted_df["Team"], sorted_df["PredictedYellowCards"], label="Yellow Cards", color="gold")
    ax.bar(sorted_df["Team"], sorted_df["PredictedRedCards"],
           bottom=sorted_df["PredictedYellowCards"], label="Red Cards", color="red")
    ax.set_ylabel("Predicted Cards")
    ax.set_title("Predicted Yellow & Red Cards per Team")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

def plot_psychological_match_predictions(psych_preds):
    """
    function used to visualize predicted win probabilities for remaining fixtures as a stacked bar chart
    Args:
        - psych_preds: output DataFrame from psychological match predictor
    Returns:
        None
"""
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [f"{row['HomeTeam']} vs {row['AwayTeam']}" for _, row in psych_preds.iterrows()]
    x = range(len(labels))
    ax.barh(x, psych_preds["AwayWinProb"], label="Away Win", color="lightcoral")
    ax.barh(x, psych_preds["DrawProb"],
            left=psych_preds["AwayWinProb"], label="Draw", color="gray")
    ax.barh(x, psych_preds["HomeWinProb"],
            left=psych_preds["AwayWinProb"] + psych_preds["DrawProb"], label="Home Win", color="lightgreen")
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Win Probability")
    ax.set_title("Psychological Match Outcome Predictions")
    ax.legend()
    st.pyplot(fig)




def train_expected_goals_model(df, season, cutoff_week=34):
    """
    function used to train a regression model to predict expected goals per team using recent form and rolling goal stats
    Args:
        - df: match-level data
        - season: current season to exclude
        - cutoff_week: matchweek cutoff for training
    Returns:
        - model: trained XGBRegressor
        - scaler: fitted feature scaler
"""

    df = df[(df["Season"] != season) & (df["MatchWeek"] <= cutoff_week)]
    df = add_match_points(df)

    rolling = compute_rolling_goal_stats(df)
    recent_form = compute_recent_form(df)
    
    features_df = rolling.merge(recent_form, on=["Season", "Team"], how="left").dropna()
    features_df = features_df.rename(columns={"AvgGoalsScored": "xG", "RecentFormPPG": "Form"})

    X = features_df[["xG", "Form"]]
    y = features_df["xG"]

    scaler = StandardScaler()
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(scaler.fit_transform(X), y)

    return model, scaler







def train_over_under_model(df, season, cutoff_week=34, threshold=2.5):
    """
    function used to train a classifier for predicting if matches will go over 2.5 goals based on performance features
    Args:
        - df: historical match data
        - season: current season to exclude from training
        - cutoff_week: matchweek cutoff
        - threshold: goals threshold for over/under
    Returns:
        - model: trained XGBClassifier
        - scaler: fitted feature scaler
        - features: list of features used for prediction
"""
    df = df[df["Season"] != season]
    df = df[df["MatchWeek"] <= cutoff_week]
    df = add_match_points(df)

    # Ensure required columns
    required_cols = [
        "FTHG", "FTAG", "HomeTeamShots", "AwayTeamShots",
        "HomeTeamShotsOnTarget", "AwayTeamShotsOnTarget",
        "HomeTeamCorners", "AwayTeamCorners",
        "B365Over2.5Goals", "B365Under2.5Goals"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Target variable
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["Over"] = (df["TotalGoals"] > threshold).astype(int)

    # Feature engineering
    df["AvgShotsPerGame"] = (df["HomeTeamShots"] + df["AwayTeamShots"]) / 2
    df["AvgShotsOnTargetPerGame"] = (df["HomeTeamShotsOnTarget"] + df["AwayTeamShotsOnTarget"]) / 2
    df["AvgCornersPerGame"] = (df["HomeTeamCorners"] + df["AwayTeamCorners"]) / 2
    #df["AvgGoalsPerGame"] = df["TotalGoals"]
    df["RecentFormGap"] = 0  # Placeholder — you can compute actual form diffs if needed
    df["xG_Gap"] = df["FTHG"] - df["FTAG"]  # Proxy for goal difference (xG substitute)

    # Define features
    
    features = [
        "AvgShotsPerGame", "AvgShotsOnTargetPerGame", "AvgCornersPerGame",
        "B365Over2.5Goals", "B365Under2.5Goals",
        "RecentFormGap", "xG_Gap"
    ]

    X = df[features].fillna(0)
    y = df["Over"]

    # Train/Validation Split ✅
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale + Train
    scaler = StandardScaler()
    model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric="logloss")
    model.fit(scaler.fit_transform(X_train), y_train)

    # Evaluate
    y_pred = model.predict(scaler.transform(X_val))
    y_prob = model.predict_proba(scaler.transform(X_val))[:, 1]

    acc = accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_prob)

    st.caption(f"⚽ Over {threshold} Goals Model — Accuracy: {acc*100:.1f}%, Log Loss: {loss:.4f}")
    return model, scaler, features


def train_expected_corners_model(df, season, cutoff_week=34):
    """
    function used to train a regression model to predict average corners per team based on form and historical stats
    Args:
        - df: historical match data
        - season: current season to exclude
        - cutoff_week: matchweek cutoff
    Returns:
        - model: trained XGBRegressor
        - scaler: fitted feature scaler
"""
    df = df[(df["Season"] != season) & (df["MatchWeek"] <= cutoff_week)]

    df["HC"] = df.get("HC", 0)
    df["AC"] = df.get("AC", 0)

    home_corners = df.groupby("HomeTeam")["HC"].mean()
    away_corners = df.groupby("AwayTeam")["AC"].mean()

    team_corners = pd.concat([
        home_corners.rename("Corners"),
        away_corners.rename("Corners")
    ]).groupby(level=0).mean().reset_index().rename(columns={"index": "Team"})

    recent_form = compute_recent_form(df)
    features_df = team_corners.merge(recent_form, on="Team", how="left").dropna()

    X = features_df[["Corners", "RecentFormPPG"]]
    y = features_df["Corners"]

    scaler = StandardScaler()
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(scaler.fit_transform(X), y)

    return model, scaler



def predict_upcoming_goals(model, scaler, remaining_df, goal_stats):
    """
    function used to predict expected home and away goals for upcoming fixtures
    Args:
        - model: trained goal prediction model
        - scaler: feature scaler
        - remaining_df: fixture list
        - goal_stats: dictionary of per-team stats
    Returns:
        - pd.DataFrame: fixtures with expected home and away goals
"""
    df = remaining_df.copy()
    home_preds, away_preds = [], []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_stats = goal_stats.get(home, {"AvgGoalsScored": 1.3, "RecentFormPPG": 1.3})
        away_stats = goal_stats.get(away, {"AvgGoalsScored": 1.1, "RecentFormPPG": 1.3})

        x_home = pd.DataFrame([{
            "xG": home_stats["AvgGoalsScored"],
            "Form": home_stats["RecentFormPPG"]
        }])
        x_away = pd.DataFrame([{
            "xG": away_stats["AvgGoalsScored"],
            "Form": away_stats["RecentFormPPG"]
        }])

        home_pred = model.predict(scaler.transform(x_home))[0]
        away_pred = model.predict(scaler.transform(x_away))[0]

        home_preds.append(round(home_pred, 2))
        away_preds.append(round(away_pred, 2))

    df["ExpectedHomeGoals"] = home_preds
    df["ExpectedAwayGoals"] = away_preds
    return df



def predict_over_under_probs(model, scaler, upcoming_df, goal_stats_df, features, threshold=2.5):
    """
    function used to predict probability of over 2.5 goals for upcoming matches using goal stats and engineered features
    Args:
        - model: trained over/under classifier
        - scaler: feature scaler
        - upcoming_df: DataFrame of upcoming matches
        - goal_stats_df: DataFrame with expected goal metrics
        - features: list of engineered features used for model input
        - threshold: goal threshold (default 2.5)
    Returns:
        - pd.DataFrame: match predictions with probabilities and expected goals
"""
    df = upcoming_df.copy()

    # Merge expected goals and form from goal_stats_df
    merged = df.merge(goal_stats_df, left_on="HomeTeam", right_on="Team", suffixes=("", "_Home"))
    merged = merged.merge(goal_stats_df, left_on="AwayTeam", right_on="Team", suffixes=("", "_Away"))

    # Compute engineered features
    merged["xG_Home"] = merged["AvgGoalsScored"]
    merged["xG_Away"] = merged["AvgGoalsConceded_Away"]
    merged["RecentFormPPG_Home"] = merged["RecentFormPPG"]
    merged["RecentFormPPG_Away"] = merged["RecentFormPPG_Away"]

    for col in ["ExpectedShots_Home", "ExpectedShots_Away", "ExpectedShotsOnTarget_Home", "ExpectedShotsOnTarget_Away", "ExpectedCorners_Home", "ExpectedCorners_Away"]:
        if col not in merged.columns:
            merged[col] = 10 if "Shots" in col else 5

    merged["AvgShotsPerGame"] = (merged["ExpectedShots_Home"] + merged["ExpectedShots_Away"]) / 2
    merged["AvgShotsOnTargetPerGame"] = (merged["ExpectedShotsOnTarget_Home"] + merged["ExpectedShotsOnTarget_Away"]) / 2
    merged["AvgCornersPerGame"] = (merged["ExpectedCorners_Home"] + merged["ExpectedCorners_Away"]) / 2
    merged["AvgGoalsPerGame"] = merged["xG_Home"] + merged["xG_Away"]

    if "B365Over2.5Goals" not in df.columns:
        merged["B365Over2.5Goals"] = 2.5
        merged["B365Under2.5Goals"] = 1.6
    else:
        merged["B365Over2.5Goals"] = df["B365Over2.5Goals"].values
        merged["B365Under2.5Goals"] = df["B365Under2.5Goals"].values

    # New engineered features
    merged["RecentFormGap"] = merged["RecentFormPPG_Home"] - merged["RecentFormPPG_Away"]
    merged["xG_Gap"] = merged["xG_Home"] - merged["xG_Away"]

    # Model prediction
    X = merged[features].fillna(0)
    probs = model.predict_proba(scaler.transform(X))[:, 1]

    # Attach predictions to the same merged DataFrame
    merged["Over2.5_Prob"] = np.round(probs * 100, 1)
    merged["Over2.5Prediction"] = (probs > 0.5).astype(int)

    # Return relevant columns only
    return merged[[
        "Date", "HomeTeam", "AwayTeam",
        "Over2.5_Prob", "Over2.5Prediction",
        "ExpectedHomeGoals", "ExpectedAwayGoals"
    ]]

def predict_upcoming_corners(model, scaler, df, team_corners):
    """
    function used to predict expected corners for upcoming matches per team using trained regression model
    Args:
        - model: trained XGBRegressor
        - scaler: feature scaler
        - df: DataFrame of upcoming matches
        - team_corners: dictionary of per-team corner stats
    Returns:
        - pd.DataFrame: upcoming matches with expected home and away corners
"""
    df = df.copy()
    home_preds, away_preds = [], []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_stats = team_corners.get(home, {"Corners": 5, "RecentFormPPG": 1.3})
        away_stats = team_corners.get(away, {"Corners": 5, "RecentFormPPG": 1.3})

        x_home = pd.DataFrame([{
            "Corners": home_stats["Corners"],
            "RecentFormPPG": home_stats["RecentFormPPG"]
        }])
        x_away = pd.DataFrame([{
            "Corners": away_stats["Corners"],
            "RecentFormPPG": away_stats["RecentFormPPG"]
        }])

        home_pred = model.predict(scaler.transform(x_home))[0]
        away_pred = model.predict(scaler.transform(x_away))[0]

        home_preds.append(round(home_pred, 1))
        away_preds.append(round(away_pred, 1))

    df["ExpectedHomeCorners"] = home_preds
    df["ExpectedAwayCorners"] = away_preds
    df["TotalExpectedCorners"] = df["ExpectedHomeCorners"] + df["ExpectedAwayCorners"]

    return df

def plot_expected_goals_chart(df):
    """
    function used to create a bar chart of expected goals per match using match labels
    Args:
        - df: DataFrame containing expected goals per team
    Returns:
        - matplotlib.figure.Figure: figure object for rendering
"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    match_labels = [f"{row['HomeTeam']} vs {row['AwayTeam']}" for _, row in df.iterrows()]
    x = range(len(df))
    home_goals = df["ExpectedHomeGoals"]
    away_goals = df["ExpectedAwayGoals"]

    ax.bar(x, home_goals, label="Home Goals", color="skyblue")
    ax.bar(x, away_goals, bottom=home_goals, label="Away Goals", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(match_labels, rotation=45, ha="right")
    ax.set_ylabel("Expected Goals")
    ax.set_title("Expected Goals per Match")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_expected_corners_chart(df):
    """
    function used to create a bar chart of expected corners per match for home and away teams
    Args:
        - df: DataFrame containing expected corners
    Returns:
        - matplotlib.figure.Figure: figure object for rendering
"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    match_labels = [f"{row['HomeTeam']} vs {row['AwayTeam']}" for _, row in df.iterrows()]
    x = range(len(df))
    home_corners = df["ExpectedHomeCorners"]
    away_corners = df["ExpectedAwayCorners"]

    ax.bar(x, home_corners, label="Home Corners", color="mediumseagreen")
    ax.bar(x, away_corners, bottom=home_corners, label="Away Corners", color="lightcoral")

    ax.set_xticks(x)
    ax.set_xticklabels(match_labels, rotation=45, ha="right")
    ax.set_ylabel("Expected Corners")
    ax.set_title("Expected Corners per Match")
    ax.legend()
    plt.tight_layout()
    return fig