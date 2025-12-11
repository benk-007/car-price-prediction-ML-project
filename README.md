# 🚗 Car Price Prediction – Machine Learning Project

This repository presents a complete Machine Learning workflow designed to predict used car prices based on technical specifications, vehicle condition, and historical data.  
The project covers data preprocessing, feature engineering, model training, performance evaluation, and explainability using SHAP.

---

## 📌 Project Objectives

The goal of this project is to build an end-to-end predictive system capable of estimating the price of a used car.  
The workflow includes:

- ✔ Data cleaning & preparation  
- ✔ Feature engineering (Car Age, km/year, ratios)  
- ✔ Exploratory data analysis (EDA)  
- ✔ Training multiple ML models  
- ✔ Model comparison through metrics (RMSE, MAE, R²)  
- ✔ Explainability using SHAP values  
- ✔ Exporting the final model + scaler  

---

## 📂 Repository Structure

```
car-price-prediction-ML-project/
├── car_price_prediction.ipynb   # Main ML notebook
├── car_price_dataset.csv         # Raw dataset used for training
├── gb_model.pkl                  # Final Gradient Boosting model (exported)
├── scaler.pkl                    # Standard scaler used during preprocessing
├── .gitignore
└── README.md                     # Documentation 
```

---

## 🛠 Tech Stack

- **Python 3.10+**
- **NumPy, Pandas** – Data manipulation  
- **Matplotlib, Seaborn** – Visualization  
- **Scikit-learn** – ML Models  
- **SHAP** – Explainable AI  
- **Joblib** – Model serialization  

---

## 🚀 Getting Started

### 🔹 Installation

Install required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap joblib
```

### 📝 Running the Notebook (Optional)

To run the notebook locally:

```bash
pip install jupyter
jupyter notebook
```

---

## 📊 Workflow Summary

### 1️⃣ Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Splitting dataset into train/test sets

### 2️⃣ Feature Engineering
- `Car_Age = 2025 - Year`
- `Km_per_Year = Mileage / Car_Age`
- `Engine_per_Door = Engine_Size / Number_of_Doors`

### 3️⃣ Model Training

**Models tested:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor *(Best model)*

**Evaluation metrics:**
- RMSE
- MAE
- R² Score

### 4️⃣ Explainability (XAI)

Using **SHAP** to understand:
- Global feature importance
- Local predictions
- Which features drive price up or down

**Examples of important features:**

| Feature       | Influence |
|---------------|-----------|
| Car Age       | 🔥 High   |
| Mileage       | 🔥 High   |
| Engine Size   | Medium    |
| Brand / Model | Medium    |

---

## 📁 Outputs

The project exports two artifacts:

- `gb_model.pkl` → Trained Gradient Boosting model
- `scaler.pkl` → Normalization scaler used in the preprocessing pipeline

These files can be integrated into a Flask or FastAPI application for real-time predictions.

---

## 🧪 How to Use the Saved Model

Example Python script:

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input row
sample = pd.DataFrame([{
    "Year": 2018,
    "Mileage": 85000,
    "Engine_Size": 1.6,
    "Doors": 4,
    "Brand_Toyota": 1,
    "Transmission_Automatic": 1,
    # etc...
}])

# Scale and predict
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("Predicted Price:", prediction[0])
```

---

## 👨‍💻 Author

**Mokhtar BENKIRANE**  
Machine Learning & Data Science Enthusiast  
📍 Morocco

If you find this project useful, ⭐ feel free to star the repository!

---

## 📣 Future Improvements

- ✔ Deploy prediction API using Flask or FastAPI
- ✔ Build an interactive UI using Streamlit
- ✔ Add hyperparameter tuning (Random Search, Optuna)
- ✔ Add CI/CD pipeline and unit tests
