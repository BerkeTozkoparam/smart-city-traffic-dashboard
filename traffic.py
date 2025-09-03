# traffic_dashboard.py
# Smart City Traffic Volume Prediction Dashboard
# Author: Berke Baran Tozkoparan

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# -----------------------
# 1. Load trained model & feature names
# -----------------------
# NOT: Bu dosya, modeli eğitirken kaydedilen (model, feature_names) tuple'ını kullanır
model, feature_names = joblib.load("traffic_model_rf.pkl")

# -----------------------
# 2. Load dataset
# -----------------------
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

# -----------------------
# 3. Initialize Dash app
# -----------------------
app = Dash(__name__)
app.title = "Traffic Volume Dashboard"

# -----------------------
# 4. Layout
# -----------------------
app.layout = html.Div([
    html.H1("Smart City Traffic Volume Prediction Dashboard"),

    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=df["date_time"].min().date(),
            max_date_allowed=df["date_time"].max().date(),
            start_date=df["date_time"].min().date(),
            end_date=df["date_time"].max().date()
        )
    ], style={'marginBottom': 20, 'marginTop': 20}),

    html.Div([
        html.H3("Model Performance Metrics"),
        html.Div(id='metrics-output', style={'fontSize': 18})
    ], style={'marginBottom': 20}),

    dcc.Graph(id='traffic-graph'),
    dcc.Graph(id='feature-importance-graph')
])

# -----------------------
# 5. Callback
# -----------------------
@app.callback(
    Output('metrics-output', 'children'),
    Output('traffic-graph', 'figure'),
    Output('feature-importance-graph', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_dashboard(start_date, end_date):
    # Filter by date
    df_filtered = df[(df["date_time"] >= start_date) & (df["date_time"] <= end_date)].copy()

    # Feature engineering
    df_filtered["hour"] = df_filtered["date_time"].dt.hour
    df_filtered["dayofweek"] = df_filtered["date_time"].dt.dayofweek
    df_filtered["month"] = df_filtered["date_time"].dt.month
    df_filtered["is_weekend"] = df_filtered["dayofweek"].isin([5,6]).astype(int)
    df_filtered["rush_hour"] = df_filtered["hour"].isin([7,8,9,16,17,18,19]).astype(int)
    df_filtered["temp_c"] = df_filtered["temp"] - 273.15
    df_filtered["is_rainy"] = (df_filtered["rain_1h"] > 0).astype(int)
    df_filtered["is_snowy"] = (df_filtered["snow_1h"] > 0).astype(int)
    df_filtered["is_cloudy"] = (df_filtered["clouds_all"] > 50).astype(int)

    # One-hot encode categorical features
    df_filtered = pd.get_dummies(df_filtered, columns=["holiday","weather_main"], drop_first=True)

    # Lag / rolling features
    df_filtered = df_filtered.sort_values("date_time").reset_index(drop=True)
    df_filtered["lag_1"] = df_filtered["traffic_volume"].shift(1)
    df_filtered["lag_3"] = df_filtered["traffic_volume"].shift(3)
    df_filtered["roll_mean_3"] = df_filtered["traffic_volume"].rolling(3).mean()
    df_filtered["roll_std_3"] = df_filtered["traffic_volume"].rolling(3).std()
    df_filtered = df_filtered.dropna().reset_index(drop=True)

    # Align with training features
    X = df_filtered.reindex(columns=feature_names, fill_value=0)
    y_true = df_filtered["traffic_volume"]

    # Predict
    y_pred = model.predict(X)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics_text = f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}"

    # Traffic figure
    traffic_fig = go.Figure()
    traffic_fig.add_trace(go.Scatter(x=df_filtered["date_time"], y=y_true,
                                     mode='lines+markers', name="Actual"))
    traffic_fig.add_trace(go.Scatter(x=df_filtered["date_time"], y=y_pred,
                                     mode='lines+markers', name="Predicted"))
    traffic_fig.update_layout(title="Actual vs Predicted Traffic Volume",
                              xaxis_title="DateTime",
                              yaxis_title="Traffic Volume")

    # Feature importance figure
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    feat_fig = go.Figure([go.Bar(x=importances.values, y=importances.index, orientation='h')])
    feat_fig.update_layout(title="Top 15 Feature Importances",
                           xaxis_title="Importance",
                           yaxis_title="Feature")

    return metrics_text, traffic_fig, feat_fig

# -----------------------
# 6. Run server
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)
