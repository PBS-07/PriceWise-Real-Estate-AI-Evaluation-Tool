# PriceWise — Real Estate AI Evaluation Tool

> **Predict smarter. Invest better.**

PriceWise is a data-driven web application to evaluate residential property prices in **Bengaluru** based on location, size, bedroom count, bathrooms, and amenity attributes. Powered by Machine Learning models (Random Forest, XGBoost, and Linear Regression), PriceWise delivers instant fair market estimates, confidence margins, model explainability, exploratory market analytics, locality comparisons, and interactive geospatial mapping.

---

## 🌟 Key Features

1. **🎯 PriceWise Prediction Engine**: Instant fair valuation in ₹ Lakhs & ₹ Crores with confidence intervals ($\pm$ error margin) and rate per sq. ft.
2. **🔍 Model Explainability**: Visualizes top valuation drivers (property size, Bengaluru locality premium, BHK ratio) with plain-language summaries.
3. **📊 Bengaluru EDA Dashboard**: Interactive distributions, box plots by BHK, top locality rates, and size vs. price scatter plots.
4. **⚖️ Locality Comparison Tool**: Side-by-side analysis of 2–4 Bengaluru micro-markets.
5. **🎛️ Live "What-If" Simulator**: Real-time attribute sliders with live price response curves.
6. **🗺️ Bengaluru Locality Heatmap**: Geospatial interactive dark map powered by Google Maps & Folium.
7. **⚙️ AI Engine Benchmarks**: Comparative evaluation across Random Forest, XGBoost, and Linear Regression.
8. **📄 Printable PDF Reports**: One-click generation of formal property valuation reports.

---

## 📍 Dataset & Scope

- **Scope**: Bengaluru Real Estate Market (`bengaluru_house_prices.csv`)
- **Size**: 12,513 cleaned property listings across 50+ Bengaluru localities (Whitefield, Sarjapur Road, Electronic City, Indiranagar, Koramangala, HSR Layout, Thanisandra, Yelahanka, Hebbal, etc.)

---

## 🛠️ Tech Stack

- **Frontend & App Interface**: Streamlit with custom Dark Glassmorphism CSS
- **Machine Learning**: Scikit-Learn (Random Forest, Linear Regression), XGBoost
- **Data Processing**: Pandas, NumPy, Joblib
- **Visualizations**: Plotly Express, Plotly Graph Objects, Folium (`streamlit-folium`)
- **Reporting**: ReportLab PDF Generator

---

## 🚀 Running Locally

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost plotly folium streamlit-folium reportlab joblib

# Run the PriceWise Application
python -m streamlit run app.py
```
