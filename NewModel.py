import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Import Modular Components ===
from team_models import (
    load_data_sources, compute_recent_form, build_goal_stats_df,
    train_model_on_remaining_points, build_partial_features, predict_final,
    display_psychological_table, train_discipline_risk_model, plot_discipline_predictions,
    train_psych_match_predictor, predict_remaining_matches_with_psych, plot_psychological_match_predictions,
    train_expected_goals_model, train_over_under_model, train_expected_corners_model,
    predict_upcoming_goals, predict_over_under_probs, predict_upcoming_corners
)
from player_models import (
    build_player_features, train_player_goal_model,
    predict_player_goal_output, compute_star_power_index,build_star_power_features,train_star_classifier,
    build_gk_features, train_gk_classifier,build_clutch_features_from_totals,
    train_clutch_player_model, get_latest_csv
)

# === Streamlit App Entry Point ===
def main():
    """
    function used to build and run the full Streamlit web app for Premier League prediction
    Args:
        None — data loading and model training are handled internally
    Returns:
        None — renders all visualizations and tabs for predictions, models, psychological insights, and player performance
"""
    st.set_page_config(page_title="Premier League Predictor", layout="wide")

    matches, past, remaining, latest_season = load_data_sources()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏟️ Premier League Final Points Predictor")
    with col2:
        st.metric(label="📅 Season", value=latest_season)

    st.markdown("---")

    tab1, tab4, tab5, tab6 = st.tabs([
        "🔮 Predictions", 
        # "📚 Training Summary", 
        # "🗂️ Raw Match Data", 
        "🧠 Psychological Factors",
        "📈 Match Forecasts",
        "🧑‍💼 Player Model"
    ])

    # with tab3:
    #     st.subheader(f"📂 Raw Match Data: {latest_season}")
    #     with st.expander("🔍 Show Last 10 Matches of Season"):
    #         st.dataframe(matches[matches["Season"] == latest_season]
    #                      .sort_values("Date")
    #                      .tail(10),
    #                      use_container_width=True,
    #                      hide_index=True)

    # with tab2:
    #     st.subheader("📚 Training on Past Seasons")
    #     cutoff_points = st.slider("Select Matchweek Cutoff for Final Points Model", 1, 38, value=34, key="cutoff_points")
    #     model, scaler = train_model_on_remaining_points(matches, past, remaining, latest_season, cutoff_week=cutoff_points)
    #     if model is None or scaler is None:
    #         st.warning("⚠️ Skipping prediction until model is trained.")
    #         return

    with tab1:
        st.subheader(f"📊 Predicted Final Standings — {latest_season}")
        cutoff_pred = st.slider("Matchweek Cutoff (for prediction)", 1, 38, value=34, key="cutoff_prediction")

        model_pred, scaler_pred = train_model_on_remaining_points(matches, past, remaining, latest_season, cutoff_week=cutoff_pred)
        if model_pred is not None and scaler_pred is not None:
            current_agg = build_partial_features(matches, latest_season, remaining, cutoff_week=cutoff_pred)
            if not current_agg.empty:
                predict_final(model_pred, scaler_pred, current_agg)

    with tab4:
        st.subheader("🧠 Team Psychological & Discipline Stats")
        display_psychological_table(matches, latest_season, remaining)

        st.subheader("🔮 AI-Predicted Discipline Risk")
        cutoff_discipline = st.slider("Select Matchweek Cutoff for Discipline Model", 1, 38, value=34, key="cutoff_discipline")
        model_yellow, model_red, disc_scaler = train_discipline_risk_model(matches, latest_season, cutoff_week=cutoff_discipline)

        disc_df = display_psychological_table(matches, latest_season, remaining, return_df=True)
        disc_df = disc_df.rename(columns={
            "FoulsPerGame": "Fouls",
            "YellowCardsPerGame": "YellowCards",
            "RedCardsPerGame": "RedCards"
        })

        required_features = ["Fouls", "YellowCards", "RedCards", "RecentFormPPG", "HighPressureFlag", "RefereeStrictness"]
        disc_df[required_features] = disc_df[required_features].fillna(0)
        disc_X = disc_df[required_features]
        disc_df["PredictedYellowCards"] = np.ceil(model_yellow.predict(disc_scaler.transform(disc_X))).astype(int)
        disc_df["PredictedRedCards"] = np.ceil(model_red.predict(disc_scaler.transform(disc_X))).astype(int)

        view_mode_disc = st.radio("View Mode for Discipline Prediction", ["📋 Table", "📊 Chart"], horizontal=True)
        if view_mode_disc == "📋 Table":
            st.dataframe(disc_df[["Team", "PredictedYellowCards", "PredictedRedCards"]]
                         .sort_values("PredictedYellowCards", ascending=False).reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            plot_discipline_predictions(disc_df)

        st.markdown("---")
        st.subheader("🧠 Psychological Match Outcome Predictor")
        model_psych = train_psych_match_predictor(matches, latest_season, cutoff_week=cutoff_discipline)
        psych_preds = predict_remaining_matches_with_psych(matches, remaining, model_psych, latest_season)

        view_mode_psych = st.radio("View Mode for Match Predictions", ["📋 Table", "📊 Chart"], horizontal=True)
        if view_mode_psych == "📋 Table":
            st.dataframe(psych_preds.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            plot_psychological_match_predictions(psych_preds)

        st.markdown("---")

    with tab5:
        import matplotlib.pyplot as plt

        st.subheader("⚽ Expected Goals, Over/Under & Corners")
        goal_model, goal_scaler = train_expected_goals_model(matches, latest_season, cutoff_week=cutoff_discipline)
        over_model, over_scaler, over_features = train_over_under_model(matches, latest_season, cutoff_week=cutoff_discipline)
        corner_model, corner_scaler = train_expected_corners_model(matches, latest_season, cutoff_week=cutoff_discipline)

        goal_stats_df = build_goal_stats_df(matches, latest_season)
        goal_stats_dict = goal_stats_df.set_index("Team").to_dict(orient="index")
        goal_preds = predict_upcoming_goals(goal_model, goal_scaler, remaining[remaining["Season"] == latest_season], goal_stats_dict)
        over_preds = predict_over_under_probs(over_model, over_scaler, goal_preds, goal_stats_df, features=over_features, threshold=2.5)

        home_corners = matches.groupby("HomeTeam")["HC"].mean().reset_index().rename(columns={"HomeTeam": "Team", "HC": "HomeCorners"})
        away_corners = matches.groupby("AwayTeam")["AC"].mean().reset_index().rename(columns={"AwayTeam": "Team", "AC": "AwayCorners"})
        team_corners_df = pd.merge(home_corners, away_corners, on="Team", how="outer").fillna(0)
        team_corners_df["Corners"] = (team_corners_df["HomeCorners"] + team_corners_df["AwayCorners"]) / 2
        form_df = compute_recent_form(matches)[compute_recent_form(matches)["Season"] == latest_season]
        team_corners_df = team_corners_df.merge(form_df[["Team", "RecentFormPPG"]], on="Team", how="left").fillna(0)
        team_corners_dict = team_corners_df[["Team", "Corners", "RecentFormPPG"]].set_index("Team").to_dict(orient="index")
        corner_preds = predict_upcoming_corners(corner_model, corner_scaler, over_preds, team_corners_dict)

        view_mode_over = st.radio("View Mode for Over 2.5 Goals Predictions", ["📋 Table", "📊 Chart"], horizontal=True)
        if view_mode_over == "📋 Table":
            st.dataframe(
                corner_preds[[
                    "Date", "HomeTeam", "AwayTeam",
                    "ExpectedHomeGoals", "ExpectedAwayGoals",
                    "Over2.5_Prob", "Over2.5Prediction",
                    "ExpectedHomeCorners", "ExpectedAwayCorners", "TotalExpectedCorners"
                ]].sort_values("ExpectedHomeGoals", ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
        else:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            labels = [f"{row['HomeTeam']} vs {row['AwayTeam']}" for _, row in corner_preds.iterrows()]
            ax1.barh(labels, corner_preds["Over2.5_Prob"], color="skyblue")
            ax1.set_xlabel("Probability (%)")
            ax1.set_title("Over 2.5 Goal Predictions")
            ax1.invert_yaxis()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.barh(labels, corner_preds["TotalExpectedCorners"], color="orange")
            ax2.set_xlabel("Expected Corners")
            ax2.set_title("Total Expected Corners Per Match")
            ax2.invert_yaxis()
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(12, 6))
            width = 0.35
            x = np.arange(len(labels))
            ax3.bar(x - width/2, corner_preds["ExpectedHomeGoals"], width, label="Home Goals", color="lightgreen")
            ax3.bar(x + width/2, corner_preds["ExpectedAwayGoals"], width, label="Away Goals", color="salmon")
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels, rotation=45, ha="right")
            ax3.set_ylabel("Expected Goals")
            ax3.set_title("Expected Home vs Away Goals")
            ax3.legend()
            st.pyplot(fig3)

    with tab6:
        st.subheader("🧑‍💼 Player Performance Models")

        try:
            import matplotlib.pyplot as plt
            # === Goal Model ===
            st.markdown("### ⚽ Goal Prediction Model (Per 90 Minutes)")
            feature_df = build_player_features(debug=False)
            if feature_df.empty:
                st.warning("⚠️ No players met the filtering criteria.")
            else:
                model, scaler, mae, r2 = train_player_goal_model(feature_df)
                preds_df = predict_player_goal_output(model, scaler, feature_df)

                st.caption(f"🎯 Model Performance — MAE: {mae:.3f} | R²: {r2:.3f}")

                view_mode_goals = st.radio("View Mode for Goal Prediction", ["📋 Table", "📊 Chart"], horizontal=True)
                if view_mode_goals == "📋 Table":
                    st.dataframe(
                        preds_df.sort_values("PredictedGoalsPer90", ascending=False).head(15),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    import matplotlib.pyplot as plt
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    top_preds = preds_df.sort_values("PredictedGoalsPer90", ascending=False).head(15)
                    ax1.barh(top_preds["Player"], top_preds["PredictedGoalsPer90"], color="green")
                    ax1.invert_yaxis()
                    ax1.set_xlabel("Predicted Goals Per 90")
                    ax1.set_title("Top Predicted Goal Scorers")
                    st.pyplot(fig1)

            # === SPI Model ===
            st.markdown("---")
            st.markdown("### 🌟 Star Power Index")

            spi_df_raw = build_star_power_features()
            spi_df = compute_star_power_index(spi_df_raw)

            # Calculate pseudo performance metrics
            from sklearn.metrics import mean_absolute_error, r2_score
            composite_target = spi_df["GoalsPer90"] + spi_df["AssistsPer90"]
            spi_mae = mean_absolute_error(composite_target, spi_df["StarScore"])
            spi_r2 = r2_score(composite_target, spi_df["StarScore"])

            st.markdown("##### 📊 Star Power Index Summary")
            st.write(f"📦 Total players rated: **{len(spi_df):,}**")
            st.write(f"⭐ Max StarScore: **{spi_df['StarScore'].max():.2f}**")
            st.write(f"🔥 Median: **{spi_df['StarScore'].median():.2f}**")
            st.write(f"🧠 Avg Score: **{spi_df['StarScore'].mean():.2f}**")

            view_mode_spi = st.radio("View Mode for Star Power Index", ["📋 Table", "📊 Chart"], horizontal=True)

            if view_mode_spi == "📋 Table":
                st.dataframe(spi_df[[
                    "Player", "Squad", "StarScore",
                    "GoalsPer90", "AssistsPer90", "xGPer90", "xAPer90",
                    "KP", "GCA90", "Touches", "CrdYPer90", "CrdRPer90"
                ]].head(15), use_container_width=True, hide_index=True)
            else:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                top_spi = spi_df.sort_values("StarScore", ascending=False).head(15)
                ax2.barh(top_spi["Player"], top_spi["StarScore"], color="purple")
                ax2.invert_yaxis()
                ax2.set_xlabel("Star Power Index")
                ax2.set_title("Top All-Round Star Players")
                st.pyplot(fig2)
            
            # === ML-Based Star Player Classifier ===
            st.markdown("---")
            st.markdown("### 🤖 Star Classifier (ML Model)")

            star_df, auc, precision, recall, f1 = train_star_classifier()

            st.caption(
                f"🤖 Model Performance — AUC: {auc:.3f} | "
                f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
            )

            view_mode_star = st.radio("View Mode for Star Classifier", ["📋 Table", "📊 Chart"], horizontal=True)

            if view_mode_star == "📋 Table":
                st.dataframe(
                    star_df[[
                        "Player", "Squad", "StarProb", "StarPrediction",
                        "GoalsPer90", "AssistsPer90", "xGPer90", "xAPer90", "KP"
                    ]].head(15),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                top_star = star_df.sort_values("StarProb", ascending=False).head(15)
                ax.barh(top_star["Player"], top_star["StarProb"], color="gold")
                ax.set_xlabel("Predicted Star Probability")
                ax.set_title("Top Predicted Star Players (ML)")
                ax.invert_yaxis()
                st.pyplot(fig)
            
             # === GK Classifier ===
            st.markdown("---")
            st.markdown("### 🧤 Goalkeeper Star Classifier (ML)")

            gk_df, gk_auc, gk_precision, gk_recall, gk_f1 = train_gk_classifier()

            st.caption(f"🤖 Model — AUC: {gk_auc:.3f} | Precision: {gk_precision:.3f} | Recall: {gk_recall:.3f} | F1: {gk_f1:.3f}")

            view_mode_gk = st.radio("View Mode for GK Classifier", ["📋 Table", "📊 Chart"], horizontal=True)

            if view_mode_gk == "📋 Table":
                st.dataframe(
                    gk_df[[
                        "Player", "Squad", "GA", "PSxG", "GoalsPrevented", "TopGK_Prob", "TopGK_Pred",
                        "PSxG/SoT", "Stp%", "#OPA/90", "AvgDist", "Launch%", "Cmp%"
                    ]].head(15),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                top_gk = gk_df.sort_values("TopGK_Prob", ascending=False).head(15)
                ax.barh(top_gk["Player"], top_gk["TopGK_Prob"], color="teal")
                ax.set_xlabel("Probability of Top GK")
                ax.set_title("Top Goalkeepers (ML Classifier)")
                ax.invert_yaxis()
                st.pyplot(fig)



             # === Clutch Player Classifier ===
            st.markdown("---")
            st.markdown("### 🔥 Clutch Player Classifier (High-Pressure AI)")

            try:
                # Load base data
                standard_df = get_latest_csv("Standard Stats")
                misc_df = get_latest_csv("Miscellaneous Stats")
                pressure_df = display_psychological_table(matches, latest_season, remaining, return_df=True)

                # Build features
                clutch_df = build_clutch_features_from_totals(standard_df, misc_df, pressure_df)

                # Train model
                clutch_ranked, auc, precision, recall, f1 = train_clutch_player_model(clutch_df)

                st.caption(f"🎯 Clutch Model — AUC: {auc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

                view_mode_clutch = st.radio("View Mode for Clutch Classifier", ["📋 Table", "📊 Chart"], horizontal=True)

                if view_mode_clutch == "📋 Table":
                    st.dataframe(
                        clutch_ranked[[
                            "Player", "Squad", "ClutchScore", "ClutchPrediction",
                            "GoalsPer90_HP", "AssistsPer90_HP", "FoulsPer90_HP",
                            "CrdYPer90_HP", "CrdRPer90_HP", "FormInPressure"
                        ]].head(15),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    top_clutch = clutch_ranked.sort_values("ClutchScore", ascending=False).head(15)
                    ax.barh(top_clutch["Player"], top_clutch["ClutchScore"], color="orangered")
                    ax.set_xlabel("ClutchScore")
                    ax.set_title("Top High-Pressure Players (ML)")
                    ax.invert_yaxis()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Failed to run Clutch Player Classifier: {e}")



        except Exception as e:
            st.error(f"❌ Failed to load or run player models: {e}")

if __name__ == "__main__":
    main()
