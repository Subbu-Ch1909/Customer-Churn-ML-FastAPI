import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/subbu/OneDrive/Desktop/Customer Churn/data/Telco customer Churn.csv")

#Data Cleaning
print("shape:",df.shape)
print("\nColumns:",df.columns)
print("\nInfo:",df.info)
print("\nFirst Five Rows:",df.head)
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))
print(df.dtypes)
print((df["TotalCharges"]== " ").sum())
df["TotalCharges"] = df["TotalCharges"].replace(" ",np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
print("Missing TotalCharges after Conversion:", df["TotalCharges"].isna().sum())

df = df.dropna(subset = ["TotalCharges"])
print(df.dtypes)
print("\nShape after Cleaning:",df.shape)

from sklearn.model_selection import train_test_split
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"]) #Removes customerId . It is just an identifier it mightadd noise.
Y = df["Churn"]

#Startifies Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size= 0.2,
    random_state=42,
    stratify=Y
)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

print("Train Churn Ratio:", Y_train.mean())
print("Test Churn Ratio:", Y_test.mean())


#Preprocessing Step
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Identify Which columns or numerical and which are categorical
numeric_features = X_train.select_dtypes(include=["int64","float64"]).columns
categorical_features = X_train.select_dtypes(include=["object", "string"]).columns

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

#Combining both preprocessing steps into one transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

#Build the model and pipeline.
#And combine them.
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


#Train the model
model_pipeline.fit(X_train, Y_train)

print("Model Trained Successfully!")

#Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
Y_pred = model_pipeline.predict(X_test)
Y_prob = model_pipeline.predict_proba(X_test)[:,1]

#confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)

#Classification Report befor performing any tuning methods.
print("\nClassification Report of Logistic Regression Classifier:\n")
print(classification_report(Y_test, Y_pred, digits = 4))

#ROC-AUC (how well probabilities rank churners higher than non-churners)
roc_auc = roc_auc_score(Y_test, Y_prob)
print("Logistic Regression Classifier ROC-AUC:", roc_auc)



#Creating Random Forest classifier to compare the models.
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

#Create a new pipeline for the Random Forest
#We will use the same preprocessor for fair comparision
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("Classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

#Train Random Forest model.
rf_pipeline.fit(X_train, Y_train)
print("Random FOrest Classifier Model Trained Successfully")

#Generate Predictions
rf_pred = rf_pipeline.predict(X_test) #Class predictions(0 or 1)

#Probability prediction for churn (class 1)
rf_prob = rf_pipeline.predict_proba(X_test)[:,1]

#Evaluate model performance
print("\nRandom Forest Classifier Classification Report:", classification_report(Y_test,rf_pred))
print("\nRandom Forest ROC_AUC Score:", roc_auc_score(Y_test,rf_prob))


#Cross Validation
from sklearn.model_selection import cross_val_score
CV_Scores = cross_val_score(
    model_pipeline,
    X_train,
    Y_train,
    cv=5,
    scoring="roc_auc"
)
print("Cross Validation Scores:", CV_Scores)
print("Mean of ROC-AUC:", CV_Scores.mean())
print("Standard Deviation:", CV_Scores.std())


#creating custom predictions using different threshold to improve the recall score.
#Threshold tuning
from sklearn.metrics import recall_score, f1_score, precision_score
thresholds = [0.5,0.45,0.4,0.35,0.3]

for t in thresholds:
    Y_custom = (Y_prob >= t).astype(int)

    precision = precision_score(Y_test, Y_custom)
    recall = recall_score(Y_test, Y_custom)
    f1 = f1_score(Y_test, Y_custom)

    print(f"\nThreshold: {t}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")



#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
#define hyperparametr grid
param_grid ={
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}
#create Grid search
grid = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"
)
#Fit on train data
grid.fit(X_train, Y_train)
print("Best Parameters:",grid.best_params_)
print("Best Cross Validation ROC-AUC score:", grid.best_score_)

#Model Evaluation
from sklearn.metrics import roc_auc_score, classification_report
best_model = grid.best_estimator_

Y_proba_best = best_model.predict_proba(X_test)[:,1]
Y_pred_best = best_model.predict(X_test)

print("Test ROC-AUC (tuned):", roc_auc_score(Y_test, Y_proba_best))
print("Classification Report:", classification_report(Y_test,Y_pred_best))


#Feature Importance
#get Features names after preprocessing
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

#Get coefficients from logistic regression
coefficients = best_model.named_steps["classifier"].coef_[0]

#Create Dataframe
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

#sort by absolute importance
feature_importance["Abs_Coefficient"] = np.abs(feature_importance["Coefficient"])
feature_importance = feature_importance.sort_values(
    by="Abs_Coefficient",
    ascending=False
)

print("\n Top 10 most important features:\n")
print(feature_importance.head(10))

#Correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
numeric_df = df.select_dtypes(include=["int64", "float64"])
corr_matrix = numeric_df.corr()

#plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix-Numeric Features")
plt.show()


#Save the Trained pipeline
import joblib
joblib.dump(best_model, "churn_pipeline.pkl")
print("Saved: churn_pipeline.pkl")