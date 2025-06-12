 # 🥬 Palak Growth Predictor

This project is a machine learning-based predictor that estimates the leaf count (as a proxy for fresh weight) of palak (spinach) based on growth conditions like:

- NPK levels
- Soil moisture
- pH
- Temperature
- Humidity
- Day since planting

## 💡 Features

- Trained using XGBoost with polynomial features and scaled inputs
- Uses February, March field data and research-backed values (Table 1)
- Provides nutrient advice and growth validation
- MERN stack front end (in progress)

## 📁 Project Structure
├── app.py # Python script to run the model and UI (Gradio-based)
├── model.pkl # Trained XGBoost model for leaf count prediction
├── pipeline.pkl # Preprocessing pipeline (scaling + polynomial features)
├── requirements.txt # List of Python dependencies
├── sample_input.json # Example input format for testing
└── README.md # Project overview and instructions

📊 Model Metrics
MAE: 12.46
RMSE: 13.83
R²: -1.34