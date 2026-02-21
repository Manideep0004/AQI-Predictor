# AQI Predictor

A machine learning project to predict **hourly PM2.5** and derive **Air Quality Index (AQI)** categories, with an interactive **Streamlit dashboard** for date-wise analysis.

## 🔗 Live Demo

- Streamlit App: https://aqi-predictor-manideep.streamlit.app/

## 🌍 What This Project Does

- Builds lag-based time-series features from weather + historical PM2.5 data.
- Trains and compares multiple regression models.
- Selects the best-performing model (by RMSE) and saves it as `model.pkl`.
- Converts predicted PM2.5 values into AQI values and AQI categories.
- Visualizes predictions with interactive charts in Streamlit.

## 🧠 Models Trained

In `modeling.ipynb`, the following models are trained and evaluated:

1. **Linear Regression**
2. **Polynomial Regression** (degree = 2)
3. **KNN Regressor** (`n_neighbors = 5`)
4. **Support Vector Regressor (SVR)** (`kernel = "rbf"`)

> The saved production model is currently the **KNN Regressor**, stored in `model.pkl`.

## 📊 Features Used for Training

Main weather and lag-based features:

- `WS2M`, `RH2M`, `T2M`, `ALLSKY_SFC_SW_DWN`
- `PM2_5_lag1`, `PM2_5_lag3`, `PM2_5_lag6`
- `WS2M_lag1`, `WS2M_lag3`, `WS2M_lag6`

Target variable:

- `PM2_5`

## 🧮 Math Behind the Numbers

### 1) Standardization (Feature Scaling)

All input features are standardized before modeling:

$$
z = \frac{x - \mu}{\sigma}
$$

where:
- $x$ = original feature value
- $\mu$ = mean of training feature
- $\sigma$ = standard deviation of training feature

---

### 2) Evaluation Metrics

For $n$ samples, true values $y_i$ and predictions $\hat{y}_i$:

**MAE (Mean Absolute Error)**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**RMSE (Root Mean Squared Error)**

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**R² Score**

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

Model comparison is sorted by **lowest RMSE**.

---

### 3) PM2.5 → AQI Conversion Logic

AQI is computed piecewise from PM2.5 concentration based on defined breakpoints:

- $0 \le PM2.5 \le 30 \Rightarrow AQI \in [0, 50]$
- $31 \le PM2.5 \le 60 \Rightarrow AQI \in [51, 100]$
- $61 \le PM2.5 \le 90 \Rightarrow AQI \in [101, 200]$
- $91 \le PM2.5 \le 120 \Rightarrow AQI \in [201, 300]$
- $121 \le PM2.5 \le 250 \Rightarrow AQI \in [301, 400]$
- $PM2.5 > 250 \Rightarrow AQI \in [401, 500]$

Each segment uses linear interpolation:

$$
AQI = \frac{(C - C_{low})}{(C_{high} - C_{low})}(I_{high} - I_{low}) + I_{low}
$$

where $C$ is PM2.5 concentration.

## 🎨 Graphs Included

The notebook and dashboard include these visualizations:

1. **Model Comparison (RMSE bar chart)**
2. **Actual vs Predicted PM2.5 (time series)**
3. **Actual vs Predicted PM2.5 (scatter with ideal line)**
4. **Hourly AQI: Actual vs Predicted**
5. **Predicted AQI with severity bands**
6. **Prediction error by hour**
7. **Interactive AQI bar chart by category color (Streamlit)**

### Suggested Report Screenshots

If you want this README to show real chart images on GitHub, save screenshots in `assets/` and use:

```md
![Model Comparison](assets/model-comparison-rmse.png)
![AQI Severity Bands](assets/aqi-severity-bands.png)
![Hourly PM2.5](assets/hourly-pm25.png)
```

## 🗂 Project Structure

- `app.py` → Streamlit dashboard
- `modeling.ipynb` → model training, evaluation, plotting, and model export
- `data_extraction.ipynb` / `eda.IPYNB` → data collection and exploration
- `output2.csv` → training data
- `test.csv` → test/inference data
- `model.pkl` → saved trained model
- `requirements.txt` → Python dependencies

## 🚀 Run Locally

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Launch app

```bash
streamlit run app.py
```

## ✅ Dashboard Outputs

For the selected date, the app provides:

- Average predicted AQI
- Peak predicted AQI
- Average predicted PM2.5
- Full hourly prediction table
- Interactive AQI and PM2.5 charts

## 🔮 Future Improvements

- Add cross-validation and hyperparameter tuning.
- Include feature importance and SHAP-based explainability.
- Automate model retraining with new data.
- Add multi-day forecasting.

---

Built with **Python, Pandas, Scikit-learn, Plotly, and Streamlit**.
