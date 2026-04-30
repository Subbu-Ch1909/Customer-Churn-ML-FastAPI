**Customer Churn Prediction- End-to-End ML Pipeline with FastAPI Docker.**
**Overview**
This Project builds and deploys a machine learning system to predict customer churn in a telcom dataset.
The Model identifies customers at high risk of leaving, helping business take proactive retention actions such as targeted offers, outreach, or customer support.

**The System includes:**
1. Data preprocessing pipeline
2. Model Training and Validation
3. Hyperparameter Tuning
4. Threshold Optimization
5. Feature Importance and analysis
6. FastAPI based model deployment
7. Docker Containerization for portability


**Tech Stack** 
**Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn
**Deployment:** FastAPI, Uvicorn 
-Docker

**Features**
-End-to-end machine learning pipeline for churn prediction
-Data preprocessing using Scikit-learn pipelines and ColumnTransformer
-Hyperparameter tuning using GridsearchCV
-Threshold optimization to improve recall and business impact
-Model evaluation using ROC-AUC, Precision, Recall, and F1-score
-FastAPI based REST API for real time predictions
-Docker containerization for consistent and portable deployment


**Business Problem**
Customer churn directly impacts revenue.
By predicting which customers are likely to leave, business can:
1. Target high-risk customers
2. Reduce churn rate
3. Increase customer lifetime value

This model outputs:
1. Churn probability (0-1)
2. Predicted Churn(0/1) based on optimized threshold

**Dataset**
1. ~ 7,000 telecom customers
2. 19 input features
3. Target Variable: Churn (Yes/NO)
4. Churn rate : ~ 26%

**Features Categories**
1. Demographics (gender, SeniorCitizen)
2. Service information (InternetService, Contract, TechSupport)
3. Billing information (MonthlyCharges, TotalCharges, tenure)

**Machine Learning Pipeline**
*Data Preprocessing*
1. Stratifies train-test split (80/2)
2. Numeric features --> StandardScaler
3. Categorical features --> OneHotEncoder(handle_unknown = "ignore)
4. Combined using ColumnTransformer
5. Full pipeline built using sklearn.pipeline

**Model Selection**
*Models Evaluated*
1. Logistic Regression.
2. Random Forest Classifier
**Logistic Regression performed better based on ROC-AUC and interpretability.**


**Model Evaluation**
*Metrics Used*
1. Confusion Matrix
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
Test ROC-AUC: ~0.83
Cross Validation ROC-AUC (5-fold mean): ~0.846


**Threshold Optimization**
Default Threshold: 0.5
Optimized Threshold: 0.4
Improved recall to better identify high-risk customers.


**Hyperparameter Tuning**
Used GridSearchCV to tune Logistic Regression regularization strength(C).
Best Parameter
*C = 100*

**Feature Importance**
*Top Churn Drivers identified:*
1. MonthlyCharges
2. InternetService type (Fiber Optic higher churn risk)
3. tenure (longer tenure reduces churn)
4. Contract type( Two-year contracts reduce churn)
This provides actionable insights.

**Deployment**
The full preprocessing + trained model pipeline was serialized using joblib.
The model is deployes using FastAPI, exposing a REST endpoint:
**Endpoint:**
POST/predict

**Example Input**
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}

**Example Output:**
{
    "churn_probability": 0.80,
    "predicted_churn" : 1,
    "threshold used" :0.4
}


**Docker Setup**
Build Docker image
docker build -t churn-app

**Run Container**
docker run -p 8000:8000 churn-app

**How to Run Locally**
*1. create a virtual environment*
pyhton -m venv venv
venv/scripts/activate

*2. Install Dependencies*
pip install -r requirements.txt

*3. Start the API*
uvicorn app:app --reload

**Project Structure:**
.
|--app.py
|--explore.py
|--predict.py
|--churn_pipeline.pkl
|--requirements.txt
|--README.md
|--Dockerfile