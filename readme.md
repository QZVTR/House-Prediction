# Creation of Model and Basic Web Interface and API 

## 📊 Models

The following models were used and evaluated:

- ✅ XGBOOST
- Neural Network 
- Linear Regression 
- Random Forest Regressor

🏆 XGBOOST achieved lowest rmse: 47989.31 (root mean squared error) so was chosen for website. 

## 🚀 Fast API

Basic FastAPI server is used to serve the XGBoost model and predic house prices. 

### Endpoint

```
POST /predict
```

### Sample Request Body:
```
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": 1
}
```

### Sample Response:
```
{
    "model": "xgboost",
    "predicted_value": 416807.44
}
```

## 🌐 Streamlit Web App

The front-end is built using Streamlit, providing a simple UI for users to input housing features and get real-time predictions.

### Features:
- User input form for housing characterstics
- Location preview using an interactive map
- Integration with FastAPI to fetch prediction results
- Responsive and clean UI


## ✅ Requirements
- torch==2.7.1+cu118
- pandas==2.3.1
- scikit-learn==1.7.1
- seaborn==0.13.2
- matplotlib==3.10.3
- numpy==2.3.1
- streamlit==1.47.0
- fastapi==0.116.1
- joblib==1.5.1
- xgboost==3.0.2

Install dependencies:
```
pip install -r requirements.txt
```