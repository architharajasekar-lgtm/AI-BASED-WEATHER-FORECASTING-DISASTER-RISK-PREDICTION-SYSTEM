# AI-BASED-WEATHER-FORECASTING-DISASTER-RISK-PREDICTION-SYSTEM
An AI-powered system that analyzes weather data to predict future conditions and assess potential disaster risks. It leverages machine learning algorithms to provide early warnings and improve disaster preparedness
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import requests
import joblib
import shap
import os
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

FEATURE_COLS = [
    "temperature_2m",
    "rain",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure"]
TIME_COL = "time"
TIME_STEPS = 24
FORECAST_HORIZON = 1

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    values = df[FEATURE_COLS].dropna().values
    X, y = [], []
    for i in range(len(values) - TIME_STEPS - (FORECAST_HORIZON - 1)):
        past_block = values[i:i+TIME_STEPS]
        target_block = values[i+TIME_STEPS:i+TIME_STEPS+FORECAST_HORIZON]
        X.append(past_block.flatten())
        y.append(target_block.flatten())
    return np.array(X), np.array(y)

def fit_scalers(X_train, y_train, n_features):
    def _fit_X_scalers(X_flat):
        X_arr = X_flat.reshape(-1, TIME_STEPS, n_features)
        scalers = []
        for feat_idx in range(n_features):
            s = MinMaxScaler()
            s.fit(X_arr[:, :, feat_idx].reshape(-1, 1))
            scalers.append(s)
        return scalers

    X_scalers = _fit_X_scalers(X_train)
    y_scalers = [MinMaxScaler() for _ in range(n_features)]
    for fi in range(n_features):
        y_scalers[fi].fit(y_train[:, fi].reshape(-1, 1))
    return X_scalers, y_scalers

def transform_X(X_flat, scalers, n_features):
    X_arr = X_flat.reshape(-1, TIME_STEPS, n_features)
    out = np.empty_like(X_arr, dtype=float)
    for feat_idx, s in enumerate(scalers):
        col = X_arr[:, :, feat_idx].reshape(-1, 1)
        col_scaled = s.transform(col).reshape(-1, TIME_STEPS)
        out[:, :, feat_idx] = col_scaled
    return out.reshape(X_flat.shape)

def transform_y(y, y_scalers):
    return np.column_stack([y_scalers[i].transform(y[:, i].reshape(-1, 1)).ravel()
                            for i in range(len(y_scalers))])

def inverse_transform_y(y_scaled, y_scalers):
    return np.column_stack([y_scalers[i].inverse_transform(y_scaled[:, i].reshape(-1, 1)).ravel()
                            for i in range(len(y_scalers))])

# ---------------- Model Training ----------------
def train_model(model_name, X_train, y_train, X_scalers, y_scalers):
    if model_name.lower() == "xgb":
        base = XGBRegressor(objective="reg:squarederror", n_estimators=200, verbosity=0)
    elif model_name.lower() == "cat":
        base = CatBoostRegressor(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            loss_function="RMSE",
            verbose=0,
            random_seed=42
        )
    elif model_name.lower() == "lgbm":
        base = LGBMRegressor(n_estimators=200)
    else:
        raise ValueError("Invalid model_name. Choose from 'xgb', 'cat', 'lgbm'.")
    
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)

    joblib.dump({
        "model": model,
        "X_scalers": X_scalers,
        "y_scalers": y_scalers
    }, f"{model_name}_multi.joblib")
    print(f"{model_name.upper()} model saved.")
    return model

# ---------------- Model Evaluation ----------------
def evaluate_model(model, X_test, y_test, y_scalers):
    y_pred_scaled = model.predict(X_test)
    y_pred = inverse_transform_y(y_pred_scaled, y_scalers)
    
    print("Evaluation Metrics per Feature:\n")
    for i, name in enumerate(FEATURE_COLS):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        # Avoid division by zero for MAPE
        non_zero_idx = y_test[:, i] != 0
        if np.any(non_zero_idx):
            mape = np.mean(np.abs((y_test[non_zero_idx, i] - y_pred[non_zero_idx, i]) / y_test[non_zero_idx, i])) * 100
        else:
            mape = np.nan
        print(f"{name}: MSE={mse:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, MAPE={mape:.2f}%")
    return y_pred


# ---------------- Live Forecast & Data Alignment ----------------
def fetch_weather_custom(latitude, longitude):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    yesterday = now - timedelta(hours=24)
    start_date = yesterday.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,rain,relative_humidity_2m,wind_speed_10m,surface_pressure"
    )
    response = requests.get(url)
    data = response.json()

    records = []
    hours = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    for i, ts in enumerate(hours):
        if yesterday <= ts <= now:
            records.append({col: data["hourly"][col][i] for col in FEATURE_COLS})
            records[-1]["time"] = ts
    return pd.DataFrame(records).reset_index(drop=True)

def align_open_meteo_data(om_df):
    candidates = {
        "wind_speed_10m": ["windspeed_10m"],
        "relative_humidity_2m": ["relativehumidity_2m"],
    }
    out = pd.DataFrame()
    for col in FEATURE_COLS:
        if col in om_df:
            out[col] = om_df[col]
        elif col in candidates:
            for cand in candidates[col]:
                if cand in om_df:
                    out[col] = om_df[cand]
                    break
        else:
            out[col] = np.nan
    return out.tail(TIME_STEPS).ffill().bfill()

def predict_next_hour(model_bundle, last24_df):
    model = model_bundle["model"]
    X_scalers = model_bundle["X_scalers"]
    y_scalers = model_bundle["y_scalers"]

    X_live_flat = last24_df[FEATURE_COLS].values.flatten().reshape(1, -1)
    X_live_scaled = transform_X(X_live_flat, X_scalers, len(FEATURE_COLS))
    y_live_scaled = model.predict(X_live_scaled)

    y_live = np.array([y_scalers[i].inverse_transform(y_live_scaled[:, i].reshape(-1, 1)).ravel()[0]
                       for i in range(len(FEATURE_COLS))])
    return {FEATURE_COLS[i]: float(y_live[i]) for i in range(len(FEATURE_COLS))}

def fetch_weather_historical(latitude, longitude, start_year=2015, end_year=2024, save_path=None):
    import time
    all_records = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,rain,relative_humidity_2m,wind_speed_10m,surface_pressure"
        )

        try:
            response = requests.get(url)
            data = response.json()

            hours = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
            temps = data["hourly"]["temperature_2m"]
            rain = data["hourly"]["rain"]
            humidity = data["hourly"]["relative_humidity_2m"]
            windspeed = data["hourly"]["wind_speed_10m"]
            pressure = data["hourly"]["surface_pressure"]

            for i, ts in enumerate(hours):
                all_records.append({
                    "time": ts,
                    "temperature_2m": temps[i],
                    "rain": rain[i],
                    "relative_humidity_2m": humidity[i],
                    "wind_speed_10m": windspeed[i],
                    "surface_pressure": pressure[i],
                })
            print(f"Year {year} data fetched. Total records so far: {len(all_records)}")
            time.sleep(1)  # avoid rate limits

        except Exception as e:
            print(f"Failed to fetch year {year}: {e}")

    df = pd.DataFrame(all_records).sort_values("time").reset_index(drop=True)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved historical data to {save_path}")

    return df

def rule_based_extreme_events_next_hour(forecast_dict):

    temp = forecast_dict['temperature_2m']
    rain = forecast_dict['rain']
    humidity = forecast_dict['relative_humidity_2m']
    wind = forecast_dict['wind_speed_10m']
    pressure = forecast_dict['surface_pressure']

    # ---------------- Flood Risk ----------------
    flood_score = 0
    if rain > 0:
        flood_score += rain * 3             # rain intensity
    if humidity > 80:
        flood_score += (humidity - 80) * 0.5
    if wind > 15:  # strong wind + storm potential
        flood_score += 5
    if pressure < 1005:  # low pressure indicates storm
        flood_score += 5
    flood_score = min(flood_score, 100)

    if flood_score > 20:
        flood_risk = "High" if flood_score > 50 else "Moderate"
    else:
        flood_risk = "Low"

    # ---------------- Heatwave Risk ----------------
    heat_score = 0
    if temp > 35:
        heat_score += (temp - 35) * 2
    if humidity < 30:
        heat_score += (30 - humidity) * 0.5
    if wind > 20:
        heat_score += 5  # hot wind can worsen heatwave
    if pressure > 1015:
        heat_score += 2  # high pressure intensifies heat
    heat_score = min(heat_score, 100)

    if heat_score > 20:
        heatwave_risk = "High" if heat_score > 50 else "Moderate"
    else:
        heatwave_risk = "Low"

    return {
        "flood_risk": flood_risk,
        "heatwave_risk": heatwave_risk,
        "flood_score": round(flood_score, 1),
        "heat_score": round(heat_score, 1)
    }



def plot_feature_insights(df, feature_cols, n_pairplot_samples=500, save_dir=None):
    df = df.head(10000)

    # ---------------- Line plots ----------------
    plt.figure(figsize=(12, 6))
    for col in feature_cols:
        plt.plot(df.index, df[col], label=col)
    plt.title("Feature Trends Over Time")
    plt.xlabel("Index / Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "feature_trends.png"), dpi=300)
    plt.close()

    # ---------------- Histograms ----------------
    df[feature_cols].hist(bins=30, figsize=(10, 6))
    plt.suptitle("Feature Distributions (Histograms)")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "feature_histograms.png"), dpi=300)
    plt.close()

    # ---------------- Boxplots ----------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[feature_cols], orient='h')  # horizontal
    plt.title("Feature Boxplots (Outlier Detection)")
    plt.xlabel("Value")
    plt.ylabel("Features")
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "feature_boxplots_horizontal.png"), dpi=300)
    plt.show()
    plt.close()


    # ---------------- Correlation matrix ----------------
    plt.figure(figsize=(8, 6))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "feature_correlation.png"), dpi=300)
    plt.close()


import shap
import matplotlib.pyplot as plt

def explain_next_hour_prediction(model_bundle, last24_df, model_name=None):
    """
    Generate and display SHAP waterfall plots for next-hour forecasts
    for each predicted weather feature. Saves plots with model name.
    """
    model = model_bundle["model"]
    X_scalers = model_bundle["X_scalers"]

    # Flatten last 24-hour data into a single input vector
    X_live_flat = last24_df[FEATURE_COLS].values.flatten().reshape(1, -1)
    X_live_scaled = transform_X(X_live_flat, X_scalers, len(FEATURE_COLS))

    shap_values_dict = {}

    print("\n🔍 SHAP Waterfall Explanation for Next-Hour Forecasts:\n")

    for i, feat_name in enumerate(FEATURE_COLS):
        try:
            # Get sub-model for each output feature
            base_model = model.estimators_[i]

            # Create SHAP explainer
            explainer = shap.Explainer(base_model)
            shap_values = explainer(X_live_scaled)
            shap_values_dict[feat_name] = shap_values.values[0]

            # ---- Waterfall Plot ----
            print(f"\nFeature: {feat_name}")
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.title(f"SHAP Waterfall: {feat_name}")

            if IMAGES_DIR:
                # Include model name in the filename
                model_str = f"{model_name}_" if model_name else ""
                plt.savefig(os.path.join(IMAGES_DIR, f"{model_str}shap_{feat_name}.png"), dpi=300)

            plt.close()

        except Exception as e:
            print(f"⚠️ Could not explain {feat_name}: {e}")

    return shap_values_dict

# ---------------- Plotting ----------------
def plot_forecasts_combined(y_true, y_pred, feature_names, model_name=None, save_dir=None):
    n_samples = len(y_true)  # use all samples

    for i, name in enumerate(feature_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        residuals = y_t - y_p

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        title_model = f" ({model_name.upper()})" if model_name else ""
        fig.suptitle(f"{name} Forecast Analysis{title_model} - All Samples", fontsize=16)

        # True vs Predicted
        axs[0, 0].plot(np.arange(n_samples), y_t, label="True", color="blue", linewidth=2)
        axs[0, 0].plot(np.arange(n_samples), y_p, label="Predicted", color="red", linestyle="--", linewidth=2)
        axs[0, 0].set_title(f"True vs Predicted{title_model}")
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)

        # Residuals
        axs[0, 1].plot(np.arange(n_samples), residuals, color="purple")
        axs[0, 1].axhline(0, color="black", linestyle="--")
        axs[0, 1].set_title(f"Residuals{title_model}")
        axs[0, 1].grid(alpha=0.3)

        # Histogram
        axs[1, 0].hist(residuals, bins=30, color="orange", edgecolor="black")
        axs[1, 0].set_title(f"Histogram of Residuals{title_model}")
        axs[1, 0].grid(alpha=0.3)

        # Scatter
        axs[1, 1].scatter(y_t, y_p, alpha=0.6, color="green")
        axs[1, 1].plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], color="red", linestyle="--")
        axs[1, 1].set_title(f"Scatter Plot{title_model}")
        axs[1, 1].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_dir:
            fname = f"{model_name}_{name}_forecast_analysis.png" if model_name else f"{name}_forecast_analysis.png"
            plt.savefig(os.path.join(save_dir, fname), dpi=300)
        plt.close()




from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def forecast_rolling_hours(model_bundle, last24_df, hours=168):
    """
    Roll the model forward hourly using a sliding 24-hour window.
    Args:
        model_bundle: dict with keys "model", "X_scalers", "y_scalers"
        last24_df: DataFrame with TIME_STEPS rows and columns FEATURE_COLS and 'time' (datetime).
                  If 'time' missing, will create hourly timestamps ending now.
        hours: number of hours to forecast (e.g., 168 for 7 days)
    Returns:
        hourly_df: DataFrame with columns ['time', <FEATURE_COLS...>] length = hours
    """
    # clone inputs to avoid mutating caller data
    last24 = last24_df.copy().reset_index(drop=True)
    # ensure we have a time column
    if "time" not in last24.columns:
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        last24["time"] = [now - pd.Timedelta(hours=(len(last24)-1-i)) for i in range(len(last24))]
    else:
        last24["time"] = pd.to_datetime(last24["time"])

    model = model_bundle["model"]
    X_scalers = model_bundle["X_scalers"]
    y_scalers = model_bundle["y_scalers"]
    n_features = len(FEATURE_COLS)

    hourly_records = []
    window = last24[FEATURE_COLS].copy().reset_index(drop=True)

    for h in range(hours):
        # prepare flattened last 24h
        X_flat = window.values.flatten().reshape(1, -1)
        X_scaled_flat = transform_X(X_flat, X_scalers, n_features)
        # predict scaled outputs
        y_scaled = model.predict(X_scaled_flat)  # shape (1, n_features * FORECAST_HORIZON) but your setup uses single horizon
        # inverse transform
        y_pred = inverse_transform_y(y_scaled, y_scalers).ravel()[:n_features]

        # timestamp for the predicted hour: last timestamp + 1 hour
        last_time = last24["time"].iloc[-1]
        pred_time = last_time + pd.Timedelta(hours=1)

        rec = {"time": pred_time}
        for i, col in enumerate(FEATURE_COLS):
            rec[col] = float(y_pred[i])
        hourly_records.append(rec)

        # append predicted hour to window (drop oldest)
        # ensure window and last24 time columns sync
        new_row = pd.DataFrame([y_pred], columns=FEATURE_COLS)
        window = pd.concat([window.iloc[1:].reset_index(drop=True), new_row], ignore_index=True)
        # also update last24 times for next loop
        last24 = pd.concat([last24, pd.DataFrame([rec])], ignore_index=True)
        last24 = last24.reset_index(drop=True)

    hourly_df = pd.DataFrame(hourly_records)
    hourly_df["time"] = pd.to_datetime(hourly_df["time"])
    return hourly_df


def assess_disaster_risk_hourly(hour_row):
    # defensive conversions
    r = {c: float(hour_row.get(c, np.nan)) for c in FEATURE_COLS}

    # base rule labels (reuse your prior logic)
    labels = rule_based_extreme_events_next_hour(r)

    # Heuristic probability mapping for flood:
    # - heavier rain increases probability fast
    # - high humidity amplifies
    # - low surface pressure (storm) adds some weight
    rain = max(0.0, r.get("rain", 0.0))
    hum = max(0.0, r.get("relative_humidity_2m", 0.0))
    pres = r.get("surface_pressure", np.nan)

    # flood score composition (0-100)
    flood_score = 0.0
    flood_score += min(100.0, rain * 4.5)             
    flood_score += max(0.0, (hum - 60.0) * 0.6)      
    if not np.isnan(pres) and pres < 1005:
        flood_score += 8.0                           
    flood_score = float(np.clip(flood_score, 0, 100))

    temp = r.get("temperature_2m", np.nan)
    heat_score = 0.0
    if not np.isnan(temp):
        heat_score += max(0.0, (temp - 32.0) * 6.0)   
    if not np.isnan(hum):
        if hum < 30:
            heat_score += (30.0 - hum) * 0.8

        if temp and hum > 70 and temp > 33:
            heat_score += (hum - 70) * 0.5

    heat_score = float(np.clip(heat_score, 0, 100))

    return {
        "flood_prob": flood_score,
        "heatwave_prob": heat_score,
        "rules": labels
    }


def daily_risk_percentages(hourly_df, timezone=None):
    hourly = hourly_df.copy()
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.date

    flood_probs = []
    heat_probs = []
    rules_list = []

    for idx, row in hourly.iterrows():
        assessed = assess_disaster_risk_hourly(row)
        flood_probs.append(assessed["flood_prob"])
        heat_probs.append(assessed["heatwave_prob"])
        rules_list.append(assessed["rules"])

    hourly["flood_prob"] = flood_probs
    hourly["heatwave_prob"] = heat_probs

    grouped = hourly.groupby("date").agg(
        flood_pct_max=("flood_prob", "max"),
        heatwave_pct_max=("heatwave_prob", "max"),
        flood_pct_mean=("flood_prob", "mean"),
        heatwave_pct_mean=("heatwave_prob", "mean"),
        hours_count=("time", "count")
    ).reset_index()

    for col in ["flood_pct_max", "heatwave_pct_max", "flood_pct_mean", "heatwave_pct_mean"]:
        grouped[col] = grouped[col].round(2)

    return grouped


def plot_daily_risk_bars(daily_risks_df, title=None, figsize=(12,6),model_name=None, save_dir=None):
    df = daily_risks_df.copy()
    df["date_str"] = df["date"].astype(str)

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, df["flood_pct_max"], width, label="Flood % (max)")
    ax.bar(x + width/2, df["heatwave_pct_max"], width, label="Heatwave % (max)")

    ax.set_ylabel("Risk Probability (%)")
    ax.set_title(title or "Daily Natural Disaster Risk Percentages")
    ax.set_xticks(x)
    ax.set_xticklabels(df["date_str"], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"daily_risk_bars_{model_name}.png"), dpi=300)
    plt.close()



if __name__ == "__main__":
    IMAGES_DIR = r"C:\Users\LENOVO\Downloads\project" # change as needed
    os.makedirs(IMAGES_DIR, exist_ok=True) 

    LAT = float(input("Enter latitude: "))
    LON = float(input("Enter longitude: "))
    TEMP_CSV_PATH = r"D:\kerala_weather_2015_2024.csv"

    # -----------------------------
    # 1️⃣ Fetch historical data (2015–2024)
    # -----------------------------
    print("\nFetching 10-year historical data from Open-Meteo...")
    om_df_full = fetch_weather_historical(LAT, LON, 2015, 2024, save_path=TEMP_CSV_PATH)

    # -----------------------------
    # 2️⃣ Basic feature exploration
    # -----------------------------
    print("\nGenerating feature insights...")
    plot_feature_insights(om_df_full, FEATURE_COLS, save_dir=IMAGES_DIR)

    # -----------------------------
    # 3️⃣ Prepare data for ML training
    # -----------------------------
    print("\nPreparing data for training...")
    X, y = load_and_preprocess_data(TEMP_CSV_PATH)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # -----------------------------
    # 4️⃣ Fit scalers
    # -----------------------------
    print("\nFitting MinMax scalers...")
    X_scalers, y_scalers = fit_scalers(X_train, y_train, len(FEATURE_COLS))
    X_train_scaled = transform_X(X_train, X_scalers, len(FEATURE_COLS))
    y_train_scaled = transform_y(y_train, y_scalers)
    X_test_scaled = transform_X(X_test, X_scalers, len(FEATURE_COLS))

    # -----------------------------
    # 5️⃣ Train models (XGB, CatBoost, LightGBM)
    # -----------------------------
    models = {}
    for name in ["xgb", "cat", "lgbm"]:
        print(f"\nTraining {name.upper()} model...")
        models[name] = train_model(name, X_train_scaled, y_train_scaled, X_scalers, y_scalers)

    # -----------------------------
    # 6️⃣ Evaluate and visualize predictions
    # -----------------------------
    for name, model in models.items():
        print(f"\n--- {name.upper()} Evaluation ---")
        y_pred = evaluate_model(model, X_test_scaled, y_test, y_scalers)
        plot_forecasts_combined(y_test, y_pred, FEATURE_COLS,model_name=name, save_dir=IMAGES_DIR)

    # -----------------------------
    # 7️⃣ Live next-hour forecast
    # -----------------------------
    print("\nFetching last 24 hours data for live forecasting...")
    last24 = align_open_meteo_data(om_df_full.tail(TIME_STEPS))

    for name, model in models.items():
        print(f"\n====== {name.upper()} MODEL LIVE FORECAST ======")
        bundle = {"model": model, "X_scalers": X_scalers, "y_scalers": y_scalers}

        forecast = predict_next_hour(bundle, last24)
        extreme = rule_based_extreme_events_next_hour(forecast)
        shap_vals = explain_next_hour_prediction(bundle, last24, model_name = name)

        print("\nNext Hour Forecast:")
        for k, v in forecast.items():
            print(f"{k}: {v:.2f}")
        print("Extreme Event Risks:", extreme)

    # -----------------------------
    # 8️⃣ Multi-hour rolling forecast & daily disaster risk for all models
    # -----------------------------
    for name, model in models.items():
        print(f"\n===== Rolling Forecast & Risk Estimation: {model_name.upper()} =====")
        bundle = {"model": model, "X_scalers": X_scalers, "y_scalers": y_scalers}
    
        # Multi-hour rolling forecast (next 7 days / 168 hours)
        hourly_forecast_df = forecast_rolling_hours(bundle, last24, hours=168)
        print(f"{model_name.upper()} rolling forecast completed. Total records = {len(hourly_forecast_df)}")
    
        # Daily disaster risk estimation
        daily_risks_df = daily_risk_percentages(hourly_forecast_df)
    
        # Plot daily flood/heatwave risk chart
        plot_daily_risk_bars(
            daily_risks_df,
            title=f"Daily Disaster Risk Forecast ({model_name.upper()})",
            figsize=(12, 6),
            model_name = name,
            save_dir=IMAGES_DIR
        )
    
        # Display top daily risk insights
        print(f"\n{model_name.upper()} Daily Risk Summary:")
        print(daily_risks_df.tail(10))
