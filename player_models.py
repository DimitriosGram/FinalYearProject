import boto3
import pandas as pd
import io
import re
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


S3_BUCKET = "finalyearproject2025"
S3_CLIENT = boto3.client("s3")

# === Get latest file for each stat type ===
def get_latest_csv(stat_type: str):
    """
    Retrieves the latest CSV file from a specified S3 prefix and returns it as a DataFrame.

    Args:
        stat_type (str): The category of player statistics (e.g., "Standard Stats").

    Returns:
        pd.DataFrame: DataFrame containing the contents of the latest CSV file.
    """
    prefix = f"playerData/{stat_type}/"
    response = S3_CLIENT.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    csv_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".csv")]
    if not csv_files:
        return None
    latest_file = sorted(csv_files, key=lambda x: re.search(r"\d{4}-\d{2}-\d{2}", x)[0], reverse=True)[0]
    obj = S3_CLIENT.get_object(Bucket=S3_BUCKET, Key=latest_file)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), header=0)
    df.columns = df.columns.str.strip()
    return df

# === Merge all tables on shared keys ===
def load_and_merge_all_player_features():
    """
Loads all player stat tables from S3, merges them on shared keys, and returns a cleaned, combined DataFrame.

Returns:
    pd.DataFrame: Combined player statistics for valid players (90s > 3), with missing numeric values filled as 0.
"""
    # Load all stat tables
    standard = get_latest_csv("Standard Stats")
    shooting = get_latest_csv("Shooting")
    passing = get_latest_csv("Passing")
    possession = get_latest_csv("Possession")
    gca = get_latest_csv("Goal and Shot Creation")
    misc = get_latest_csv("Miscellaneous Stats")
    gk = get_latest_csv("Advanced GK Stats")

    # === Standardise merge keys ===
    keys = ["Player", "Squad", "Nation", "Age", "Born", "Pos", "90s"]

    # Start with standard
    df = standard.copy()
    for table in [shooting, passing, possession, gca, misc]:
        if table is not None:
            df = df.merge(table, on=keys, how="left", suffixes=("", "_dup"))

    # Add GK data separately (keep goalkeepers only)
    if gk is not None:
        gk["is_gk"] = True
        df = df.merge(gk[keys + ["PSxG", "PSxG/SoT", "Stp%", "AvgDist"]], on=keys, how="left")

    # Drop any duplicated columns from suffixes
    df = df.loc[:, ~df.columns.duplicated()]

    # === Convert numeric columns ===
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # === Filter valid players ===
    df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
    df = df[df["90s"] > 3]  # at least ~270 minutes

    # === Fill missing numerics with 0 ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df.reset_index(drop=True)

def build_player_features(debug=False):
    """
Constructs a filtered DataFrame for model-ready player features using stats from standard, shooting, and passing tables.

Args:
    debug (bool, optional): If True, prints debug output. Defaults to False.

Returns:
    pd.DataFrame: Filtered feature set with per-90 metrics for goal-scoring model input.
"""
    df_std = get_latest_csv("Standard Stats")
    df_shoot = get_latest_csv("Shooting")
    df_pass = get_latest_csv("Passing")

    df = df_std.merge(df_shoot, on=["Player", "Squad"], how="left", suffixes=("", "_shoot"))
    df = df.merge(df_pass, on=["Player", "Squad"], how="left", suffixes=("", "_pass"))

    df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
    df["Min"] = pd.to_numeric(df.get("Min", 0), errors="coerce")
    df["Pos"] = df.get("Pos", "").astype(str)

    # Fallback-safe goal columns
    def safe_col(df, candidates):
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(0, index=df.index)

    df["Goals"] = safe_col(df, ["Gls", "Gls_shoot"])
    df["GoalsPer90"] = df["Goals"] / df["90s"]
    df["ShotsPer90"] = safe_col(df, ["Sh", "Sh_shoot"]) / df["90s"]
    df["xGPer90"] = safe_col(df, ["xG", "xG_shoot"]) / df["90s"]
    df["xAPer90"] = safe_col(df, ["xA", "xA_pass"]) / df["90s"]

    # === Smart filters ===
    df = df[
        (df["Min"] >= 900) &
        (df["Goals"] >= 1) &
        (df["xGPer90"] >= 0.2) &
        (df["ShotsPer90"] >= 1.0) &
        (df["Pos"].str.contains("FW|MF", case=False, na=False))
    ]

    if debug:
        print("✅ Filtered down to:", len(df), "rows")
        print(df[["Player", "Squad", "Min", "GoalsPer90", "ShotsPer90", "xGPer90"]].head(10))

    return df[[
        "Player", "Squad", "GoalsPer90", "ShotsPer90", "xGPer90", "xAPer90"
    ]].dropna()






def train_player_goal_model(feature_df):
    """
Trains a regression model to predict player goals per 90 minutes using selected offensive metrics.

Args:
    feature_df (pd.DataFrame): Input features with per-90 values for shots, xG, and xA.

Returns:
    tuple: Trained model, fitted scaler, MAE, and R² score.
"""
    X = feature_df[["ShotsPer90", "xGPer90", "xAPer90"]]
    y = feature_df["GoalsPer90"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=4)
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return model, scaler, mae, r2

def predict_player_goal_output(model, scaler, feature_df):
    """
Applies the trained regression model to predict goal output per 90 minutes for each player.

Args:
    model (XGBRegressor): Trained XGBoost regression model.
    scaler (StandardScaler): Fitted scaler used during training.
    feature_df (pd.DataFrame): Feature set to make predictions on.

Returns:
    pd.DataFrame: DataFrame sorted by predicted goal output, including actual and predicted goals per 90.
"""
    X = feature_df[["ShotsPer90", "xGPer90", "xAPer90"]].fillna(0)
    feature_df["PredictedGoalsPer90"] = model.predict(scaler.transform(X))
    return feature_df.sort_values("PredictedGoalsPer90", ascending=False)[
        ["Player", "Squad", "GoalsPer90", "PredictedGoalsPer90"]
    ]


def build_star_power_features():
    """
Builds the input feature set for star player evaluation by calculating per-90 statistics from all player data tables.

Returns:
    pd.DataFrame: Filtered player DataFrame including offensive, creative, and discipline metrics per 90 minutes.
"""
    df = load_and_merge_all_player_features()

    df["GoalsPer90"] = pd.to_numeric(df.get("Gls", 0), errors="coerce") / df["90s"]
    df["AssistsPer90"] = pd.to_numeric(df.get("Ast", 0), errors="coerce") / df["90s"]
    df["xGPer90"] = pd.to_numeric(df.get("xG", 0), errors="coerce") / df["90s"]
    df["xAPer90"] = pd.to_numeric(df.get("xA", 0), errors="coerce") / df["90s"]
    df["CrdYPer90"] = pd.to_numeric(df.get("CrdY", 0), errors="coerce") / df["90s"]
    df["CrdRPer90"] = pd.to_numeric(df.get("CrdR", 0), errors="coerce") / df["90s"]
    df["FoulsPer90"] = pd.to_numeric(df.get("Fls", 0), errors="coerce") / df["90s"]
    df["KP"] = pd.to_numeric(df.get("KP", 0), errors="coerce")
    df["GCA90"] = pd.to_numeric(df.get("GCA", 0), errors="coerce") / df["90s"]
    df["SCA90"] = pd.to_numeric(df.get("SCA", 0), errors="coerce") / df["90s"]
    df["Touches"] = pd.to_numeric(df.get("Touches", 0), errors="coerce")
    df["PrgC"] = pd.to_numeric(df.get("PrgC", 0), errors="coerce")
    df["PrgP"] = pd.to_numeric(df.get("PrgP", 0), errors="coerce")
    df["PrgR"] = pd.to_numeric(df.get("PrgR", 0), errors="coerce")

    df = df[df["90s"] >= 5]  # remove low-playtime players

    return df[[
        "Player", "Squad", "GoalsPer90", "AssistsPer90", "xGPer90", "xAPer90", "KP",
        "GCA90", "SCA90", "Touches", "PrgC", "PrgP", "PrgR",
        "CrdYPer90", "CrdRPer90", "FoulsPer90"
    ]].dropna()


def compute_star_power_index(df):
    """
Computes a weighted Star Power Index score based on offensive, creative, and negative discipline metrics.

Args:
    df (pd.DataFrame): Player stats with per-90 features already computed.

Returns:
    pd.DataFrame: Sorted DataFrame including the StarScore and all input metrics.
"""
    df = df.copy()

    features_positive = [
        "GoalsPer90", "AssistsPer90", "xGPer90", "xAPer90",
        "KP", "GCA90", "SCA90", "Touches", "PrgC", "PrgP", "PrgR"
    ]
    features_negative = ["CrdYPer90", "CrdRPer90", "FoulsPer90"]

    scaler = MinMaxScaler()
    df[features_positive + features_negative] = scaler.fit_transform(df[features_positive + features_negative])

    # Weighted scoring (customizable)
    df["StarScore"] = (
        df["GoalsPer90"] * 3 +
        df["AssistsPer90"] * 2 +
        df["xGPer90"] * 2 +
        df["xAPer90"] * 2 +
        df["KP"] * 1.5 +
        df["GCA90"] * 1.5 +
        df["SCA90"] * 1.2 +
        df["Touches"] * 1 +
        df["PrgC"] * 1 +
        df["PrgP"] * 1 +
        df["PrgR"] * 1
        - df["CrdYPer90"] * 2
        - df["CrdRPer90"] * 4
        - df["FoulsPer90"] * 1
    )

    return df.sort_values("StarScore", ascending=False).reset_index(drop=True)

def train_star_classifier():
    """
Trains a classification model to identify 'star' players based on top 10% G+A per 90 minutes.

Returns:
    tuple: DataFrame with star probability/prediction, AUC score, precision, recall, and F1 score.
"""
    df = build_star_power_features().copy()

    # Target label: top 10% G+A per 90 = Star
    df["GA_per90"] = df["GoalsPer90"] + df["AssistsPer90"]
    threshold = df["GA_per90"].quantile(0.90)
    df["is_star"] = (df["GA_per90"] >= threshold).astype(int)

    features = [
        "GoalsPer90", "AssistsPer90", "xGPer90", "xAPer90", "KP",
        "GCA90", "SCA90", "Touches", "PrgC", "PrgP", "PrgR",
        "CrdYPer90", "CrdRPer90", "FoulsPer90"
    ]

    X = df[features].fillna(0)
    y = df["is_star"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    # Add predictions back to full DataFrame
    df["StarProb"] = model.predict_proba(X)[:, 1]
    df["StarPrediction"] = model.predict(X)

    return df.sort_values("StarProb", ascending=False), auc, precision, recall, f1


def build_gk_features():
    """
Loads and preprocesses goalkeeper-specific stats to compute derived features such as Goals Prevented.

Returns:
    pd.DataFrame: Goalkeeper DataFrame with cleaned, numerical features ready for model training.
"""
    df = get_latest_csv("Advanced GK Stats")
    df = df.copy()

    # Convert core features to numeric
    for col in ["GA", "PSxG", "PSxG/SoT", "Stp%", "#OPA", "#OPA/90", "AvgDist", "Launch%", "Cmp%"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0

    df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
    df = df[df["90s"] >= 5]  # Filter out low minutes

    df["GoalsPrevented"] = df["PSxG"] - df["GA"]

    return df[[
        "Player", "Squad", "GA", "PSxG", "GoalsPrevented",
        "PSxG/SoT", "Stp%", "#OPA/90", "AvgDist", "Launch%", "Cmp%"
    ]].dropna()


def train_gk_classifier():
    """
Trains a classification model to predict top-performing goalkeepers based on Goals Prevented and advanced GK metrics.

Returns:
    tuple: DataFrame with TopGK probability/prediction and performance scores (AUC, precision, recall, F1).
"""
    df = build_gk_features().copy()

    # Create target: 1 if above median GoalsPrevented
    threshold = df["GoalsPrevented"].median()
    df["is_top_gk"] = (df["GoalsPrevented"] > threshold).astype(int)

    features = ["PSxG/SoT", "Stp%", "#OPA/90", "AvgDist", "Launch%", "Cmp%"]
    X = df[features].fillna(0)
    y = df["is_top_gk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]
    auc = roc_auc_score(y_test, y_prob)

    # Apply model to all GKs
    df["TopGK_Prob"] = model.predict_proba(X)[:, 1]
    df["TopGK_Pred"] = model.predict(X)

    return df.sort_values("TopGK_Prob", ascending=False), auc, precision, recall, f1

def train_clutch_player_model(player_stats_df):
    """
Trains a classifier to predict 'clutch' players—those with top G+A per 90 during high-pressure matches.

Args:
    player_stats_df (pd.DataFrame): Player stats containing pressure-adjusted per-90 features.

Returns:
    tuple: DataFrame with ClutchScore, prediction, and model performance (AUC, precision, recall, F1).
"""
    df = player_stats_df.copy()

    # Compute total contributions
    df["GA_Per90_HP"] = df["GoalsPer90_HP"] + df["AssistsPer90_HP"]

    # Label clutch players: top 25% G+A under pressure
    threshold = df["GA_Per90_HP"].quantile(0.75)
    df["is_clutch"] = (df["GA_Per90_HP"] >= threshold).astype(int)

    features = [
        "GoalsPer90_HP", "AssistsPer90_HP", "FoulsPer90_HP",
        "CrdYPer90_HP", "CrdRPer90_HP", "FormInPressure", "PressureFlagCount"
    ]

    X = df[features].fillna(0)
    y = df["is_clutch"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    # Add predictions to full DataFrame
    df["ClutchScore"] = model.predict_proba(X)[:, 1]
    df["ClutchPrediction"] = model.predict(X)

    return df.sort_values("ClutchScore", ascending=False), auc, precision, recall, f1


def build_clutch_features_from_totals(standard_df, misc_df, team_flags_df):
    """
Builds pressure-adjusted player features (G+A, fouls, cards per 90) based on team pressure flags and match stats.

Args:
    standard_df (pd.DataFrame): Standard stat table from FBref.
    misc_df (pd.DataFrame): Miscellaneous stat table with fouls/cards.
    team_flags_df (pd.DataFrame): Team-level DataFrame with HighPressureFlag indicators.

Returns:
    pd.DataFrame: Player features adjusted for high-pressure match participation.
"""
    # Rename clearly before merge
    standard_df["Matches"] = pd.to_numeric(standard_df["Matches"], errors="coerce")
    misc_df["Matches"] = pd.to_numeric(misc_df["Matches"], errors="coerce")

    standard_df = standard_df.rename(columns={"90s": "Std90s", "Matches": "StdMatches"})
    misc_df = misc_df.rename(columns={"90s": "Misc90s", "Matches": "MiscMatches"})


    # Merge standard + misc on Player and Squad
    df = pd.merge(
        standard_df[["Player", "Squad", "Gls", "Ast", "Std90s", "StdMatches"]],
        misc_df[["Player", "Squad", "Fls", "CrdY", "CrdR", "Misc90s", "MiscMatches"]],
        on=["Player", "Squad"]
    )

    # Average shared stats
    df["90s"] = df[["Std90s", "Misc90s"]].mean(axis=1)
    df["Matches"] = df[["StdMatches", "MiscMatches"]].mean(axis=1)
    # Merge in HighPressureFlag info
    team_flags_df = team_flags_df.rename(columns={"Team": "Squad"})
    team_flags_df["PressureRatio"] = team_flags_df["HighPressureFlag"] / 38  # Assuming 38 match season

    df = df.merge(team_flags_df[["Squad", "PressureRatio"]], on="Squad", how="left")
    df["PressureRatio"] = df["PressureRatio"].fillna(0.1)  # fallback if missing

    # Estimate pressure-adjusted stats
    df["Gls_HP"] = df["Gls"] * df["PressureRatio"]
    df["Ast_HP"] = df["Ast"] * df["PressureRatio"]
    df["Fls_HP"] = df["Fls"] * df["PressureRatio"]
    df["CrdY_HP"] = df["CrdY"] * df["PressureRatio"]
    df["CrdR_HP"] = df["CrdR"] * df["PressureRatio"]
    df["90s_HP"] = df["90s"] * df["PressureRatio"]

    # Derive per-90 pressure stats
    df["GoalsPer90_HP"] = df["Gls_HP"] / df["90s_HP"]
    df["AssistsPer90_HP"] = df["Ast_HP"] / df["90s_HP"]
    df["FoulsPer90_HP"] = df["Fls_HP"] / df["90s_HP"]
    df["CrdYPer90_HP"] = df["CrdY_HP"] / df["90s_HP"]
    df["CrdRPer90_HP"] = df["CrdR_HP"] / df["90s_HP"]

    # Replace inf and nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "GoalsPer90_HP", "AssistsPer90_HP", "FoulsPer90_HP",
        "CrdYPer90_HP", "CrdRPer90_HP"
    ])

    # Add dummy FormInPressure and PressureFlagCount if unavailable
    df["FormInPressure"] = 1.3 + 0.4 * df["PressureRatio"]  # synthetic, tunable
    df["PressureFlagCount"] = 38 * df["PressureRatio"]
    df = df[df["90s_HP"] > 0]

    return df[[
        "Player", "Squad", "GoalsPer90_HP", "AssistsPer90_HP", "FoulsPer90_HP",
        "CrdYPer90_HP", "CrdRPer90_HP", "FormInPressure", "PressureFlagCount"
    ]]