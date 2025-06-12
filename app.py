import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

# --- 1. Load FEBRUARY and MARCH data ---
feb_data = pd.read_excel("FEBRUARY_UPDATED.xlsx")
mar_data = pd.read_excel("MARCH_UPDATED.xlsx")
df = pd.concat([feb_data, mar_data], ignore_index=True).dropna()

# Fix 'Day' column if string type
if df['Day'].dtype == 'object':
    df['Day'] = df['Day'].str.extract('(\d+)').astype(int)

# --- 2. Prepare Table 1 nutrient treatment data ---

# Typical environmental values (set reasonable averages)
typical_env = {
    'Temperature': 30,    # degrees Celsius
    'Humidity': 40,       # %
    'Soil Moisture': 40,  # arbitrary units
    'pH': 6.5,
    'Day': 15
}

# NPK nutrient levels (kg/ha) from Table 1 (adding a last treatment for 150 as example)
nutrient_levels = {
    'N': [0, 40, 60, 80, 100, 120],
    'P': [0, 30, 40, 50, 60, 70],
    'K': [0, 20, 30, 40, 50, 60]
}

# Corresponding fresh weights (g) from 1st cutting (Table 1)
fresh_weights = [17.27, 19.18, 20.39, 21.56, 23.25, 24.67]

# Build DataFrame from Table 1
table1_df = pd.DataFrame({
    'N': nutrient_levels['N'],
    'P': nutrient_levels['P'],
    'K': nutrient_levels['K'],
    'Temperature': typical_env['Temperature'],
    'Humidity': typical_env['Humidity'],
    'Soil Moisture': typical_env['Soil Moisture'],
    'pH': typical_env['pH'],
    'Day': typical_env['Day'],
    'Leaf Count': fresh_weights   # Using fresh weight as target proxy
})

# --- 3. Combine FEB+MAR data with Table 1 data ---
df_combined = pd.concat([df, table1_df], ignore_index=True)

# --- 4. Define features and target ---
features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'Soil Moisture', 'pH', 'Day']
target = 'Leaf Count'

X = df_combined[features]
y = df_combined[target]

# --- 5. Split train/test data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Pipeline with PolynomialFeatures, Scaler, and XGBRegressor ---
pipe_xgb = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}

# --- 7. Grid Search CV ---
grid = GridSearchCV(pipe_xgb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best XGBoost params:", grid.best_params_)

# --- 8. Train final model with early stopping on validation split ---

# Split train into train/val for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
scaler = StandardScaler()

X_tr_poly = poly.fit_transform(X_tr)
X_val_poly = poly.transform(X_val)

X_tr_scaled = scaler.fit_transform(X_tr_poly)
X_val_scaled = scaler.transform(X_val_poly)

dtrain = xgb.DMatrix(X_tr_scaled, label=y_tr)
dval = xgb.DMatrix(X_val_scaled, label=y_val)

best_params = {k.replace('xgb__', ''): v for k, v in grid.best_params_.items()}
best_params.pop('n_estimators', None)  # removed from params for xgb.train

best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42
})

evals = [(dtrain, 'train'), (dval, 'eval')]

model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=10
)

# --- 9. Save model and preprocessing objects ---
model.save_model("xgb_model.json")
joblib.dump(poly, "poly.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and preprocessors saved.")

# --- 10. Load model and preprocessors for inference example ---
model_loaded = xgb.Booster()
model_loaded.load_model("xgb_model.json")

poly_loaded = joblib.load("poly.pkl")
scaler_loaded = joblib.load("scaler.pkl")

# --- 11. Evaluate on test data ---
X_test_poly = poly_loaded.transform(X_test)
X_test_scaled = scaler_loaded.transform(X_test_poly)
dtest = xgb.DMatrix(X_test_scaled)

best_iter = model_loaded.best_iteration if model_loaded.best_iteration is not None else 0
y_pred = model_loaded.predict(dtest, iteration_range=(0, best_iter + 1))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# --- 12. Sample prediction on user input ---
user_input = {
    'N': 80,
    'P': 50,
    'K': 40,
    'Temperature': 30,
    'Humidity': 40,
    'Soil Moisture': 40,
    'pH': 6.5,
    'Day': 15
}

input_df = pd.DataFrame([user_input])
input_poly = poly_loaded.transform(input_df)
input_scaled = scaler_loaded.transform(input_poly)
dinput = xgb.DMatrix(input_scaled)

user_pred = model_loaded.predict(dinput, iteration_range=(0, best_iter + 1))
print(f"Predicted Leaf Count (Fresh Weight proxy) for user input: {user_pred[0]:.2f}")
