import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_save():
    csv_path = "cleaned_bengaluru_house_prices.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Prepare features
    # Numerical features
    num_cols = ['total_sqft', 'bhk', 'bath', 'balcony', 'has_society', 'is_ready_to_move']
    
    # Categorical encoding
    dummies_loc = pd.get_dummies(df['location'], prefix='loc', drop_first=True, dtype=int)
    dummies_area = pd.get_dummies(df['area_type'], prefix='area', drop_first=True, dtype=int)
    
    X = pd.concat([df[num_cols], dummies_area, dummies_loc], axis=1)
    y = df['price']
    
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    }
    
    metrics = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        
        metrics[name] = {
            "RMSE (₹ Lakhs)": round(rmse, 2),
            "MAE (₹ Lakhs)": round(mae, 2),
            "R2 Score": round(r2, 4)
        }
        fitted_models[name] = model

    os.makedirs("artifacts", exist_ok=True)
    
    joblib.dump(fitted_models["Random Forest"], "artifacts/model_rf.pkl")
    joblib.dump(fitted_models["Linear Regression"], "artifacts/model_lr.pkl")
    joblib.dump(fitted_models["XGBoost"], "artifacts/model_xgb.pkl")
    
    # Save feature importances from Random Forest for explainability
    rf = fitted_models["Random Forest"]
    importances = dict(zip(feature_names, rf.feature_importances_.tolist()))
    
    with open("artifacts/feature_importances.json", "w") as f:
        json.dump(importances, f)

    with open("artifacts/feature_names.json", "w") as f:
        json.dump(feature_names, f)
        
    with open("artifacts/model_metrics.json", "w") as f:
        json.dump(metrics, f)

    # Save locality list & unique values for UI dropdowns
    locations_list = sorted([str(loc) for loc in df['location'].unique()])
    area_types_list = sorted([str(a) for a in df['area_type'].unique()])
    
    metadata = {
        "locations": locations_list,
        "area_types": area_types_list,
        "total_rows": len(df),
        "avg_price_lakhs": round(float(df['price'].mean()), 2),
        "median_price_lakhs": round(float(df['price'].median()), 2),
        "avg_price_per_sqft": round(float(df['price_per_sqft'].mean()), 2)
    }
    
    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Model training on user data complete!")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train_and_save()
