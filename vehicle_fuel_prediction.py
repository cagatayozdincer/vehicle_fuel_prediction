# Vehicle Fuel Consumption Prediction Project
# Goal: Predict average fuel consumption based on vehicle specifications
# Steps: Data analysis, visualization, modeling, and performance comparison

# --- Import Required Libraries ---
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# --- Data Loading and Preparation ---
data = pd.read_csv("arac_verileri.csv")

# Rename columns for consistency
data = data.rename(columns={
    "Ortalama_yakit(l)": "target",
    "Silindir": "cylinder",
    "Beygir": "hp",
    "Agirlik_kg": "weight",
    "Hizlanma(0-100)": "acceleration",
    "Yil": "model_year",
    "Marka-Model": "brand_model"
})

# Basic data overview
print(data.head())
print("Data shape:", data.shape)
data.info()
print(data.describe())
print(data.isna().sum())

# --- Correlation Matrix Visualization ---

numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

sns.clustermap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Features")
plt.show()

# --- General Overview with Pairplot ---

sns.pairplot(data, diag_kind="kde", markers="+")
plt.show()

# --- Distribution by Cylinder Count ---

plt.figure(figsize=(6, 4))
sns.countplot(x='cylinder', data=data, palette="viridis")
plt.xlabel("Cylinder")
plt.ylabel("Vehicle Count")
plt.tight_layout()
plt.show()

# --- Outlier Detection with Boxplots ---

for col in data.columns:
    if col != "cylinder" and data[col].dtype != 'O':
        plt.figure()
        sns.boxplot(x=data[col], orient="v")
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

# --- Outlier Removal (Not Used in Final Modeling) ---

# Outlier removal was tested but excluded from the final model
# because it reduced overall performance. 
# Since the dataset is synthetic, removing outliers negatively affected model accuracy.
# The code is kept here for reference:

# threshold = 2
# numeric_cols = ["hp", "weight", "acceleration", "target"]
# for col in numeric_cols:
#     Q1 = data[col].quantile(0.25)
#     Q3 = data[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - threshold * IQR
#     upper = Q3 + threshold * IQR
#     data = data[(data[col] >= lower) & (data[col] <= upper)]

# --- Normalization of Target Variable with Log Transformation ---

# Original target distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['target'], kde=True, color="skyblue")
plt.title("Original Target Variable Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Average Fuel Consumption (L/100km)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Target distribution after log transformation
data["target"] = np.log1p(data["target"])

plt.figure(figsize=(8, 5))
sns.histplot(data['target'], kde=True, color="salmon")
plt.title("Target Distribution After Log Transformation", fontsize=14, fontweight='bold')
plt.xlabel("Log(Average Fuel Consumption + 1)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# --- Skewness Analysis of Numerical Features ---

skewed_features = numeric_data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame(skewed_features, columns=["Skewness"])
print(skewness)

# --- One Hot Encoding for Categorical Variables ---

data["cylinder"] = data["cylinder"].astype(str)
data["brand_model"] = data["brand_model"].astype(str)
data = pd.get_dummies(data)
data.head()

# --- Train-Test Split ---

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Standardization ---

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Model 1: Linear Regression ---

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression Coefficients:", lr.coef_)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression MSE: {mse:.4f}")
print(f"Linear Regression R2 Score: {r2:.4f}")

# --- Model 2: Ridge Regression with Hyperparameter Tuning ---

ridge = Ridge(max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error", refit=True)
clf.fit(X_train, y_train)

scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coefficients:", clf.best_estimator_.coef_)
ridge_best = clf.best_estimator_
print("Best Ridge Estimator:", ridge_best)

y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ridge MSE: {mse:.4f}")
print(f"Ridge R2 Score: {r2:.4f}")
print(".....................")

plt.figure(figsize=(8,5))
plt.semilogx(alphas, scores, marker='o')
plt.xlabel("Alpha")
plt.ylabel("Negative Mean Squared Error")
plt.title("Ridge Regression Cross-Validation Scores")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Model 3: Lasso Regression with Hyperparameter Tuning ---

lasso = Lasso(max_iter=1000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{"alpha": alphas}]
n_folds = 5

lasso_grid = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error", refit=True)
lasso_grid.fit(X_train, y_train)

scores = lasso_grid.cv_results_["mean_test_score"]
scores_std = lasso_grid.cv_results_["std_test_score"]

print("Lasso Coefficients:", lasso_grid.best_estimator_.coef_)
lasso_best = lasso_grid.best_estimator_
print("Best Lasso Estimator:", lasso_best)

y_pred = lasso_grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Lasso R2 Score: {r2:.4f}")
print(f"Lasso MSE: {mse:.4f}")
print(".....................")

plt.figure(figsize=(8, 5))
plt.semilogx(alphas, scores, marker='o')
plt.xlabel("Alpha")
plt.ylabel("Negative Mean Squared Error")
plt.title("Lasso Regression Cross-Validation Scores")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Model 4: ElasticNet Regression with Hyperparameter Tuning ---

param_grid = {
    "alpha": alphas,
    "l1_ratio": np.arange(0.0, 1.0, 0.05)}

elastic_net = ElasticNet(max_iter=20000)
elastic_grid = GridSearchCV(elastic_net, param_grid, cv=n_folds, scoring="neg_mean_squared_error", refit=True)
elastic_grid.fit(X_train, y_train)

print("ElasticNet Coefficients:", elastic_grid.best_estimator_.coef_)
elastic_net_best = elastic_grid.best_estimator_
print("Best ElasticNet Estimator:", elastic_net_best)

y_pred = elastic_grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"ElasticNet MSE: {mse:.4f}")
print(f"ElasticNet R2 Score: {r2:.4f}")

# --- Model 5: Random Forest ---

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regressor MSE: {mse:.4f}")
print(f"Random Forest Regressor R2 Score: {r2:.4f}")

# --- Model 6: XGBoost Regression ---

model_xgb = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=5,
    min_child_weight=4,
    subsample=0.7,
    n_estimators=1000,
    learning_rate=0.07)

model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost Regressor MSE: {mse:.4f}")
print(f"XGBoost Regressor R2 Score: {r2:.4f}")

# --- Hyperparameter Tuning and Evaluation for XGBoost Model ---

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.7, 0.8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000]}

# --- XGBoost Model and Grid Search Setup ---

xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1)

# --- Train XGBoost Model with Grid Search ---

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)

# --- Select Best XGBoost Model and Predict on Test Set ---

best_xgb = grid_search.best_estimator_

y_pred_best_xgb = best_xgb.predict(X_test)

# --- Evaluate Best XGBoost Model on Test Set ---

mse_test = mean_squared_error(y_test, y_pred_best_xgb)
r2_test = r2_score(y_test, y_pred_best_xgb)

print(f"Best XGBoost MSE on Test Set: {mse_test:.4f}")
print(f"Best XGBoost R2 on Test Set: {r2_test:.4f}")

# --- Inverse Log Transformation to Original Scale ---

y_pred_best_xgb_orig = np.expm1(y_pred_best_xgb)
y_test_orig = np.expm1(y_test)

mse_test_orig = mean_squared_error(y_test_orig, y_pred_best_xgb_orig)

print(f"Best XGBoost MSE on Test Set (Original Scale): {mse_test_orig:.4f}")

# --- Visualizing XGBoost Feature Importance ---

importances = best_xgb.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("XGBoost Feature Importance Scores", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()



# Modelleri ve isimlerini listele
models = [
    ("Linear Regression", lr),
    ("Ridge Regression", ridge_best),
    ("Lasso Regression", clf.best_estimator_),  # lasso için GridSearch sonrası model
    ("ElasticNet Regression", elastic_net_best),
    ("XGBoost Regression", best_xgb)
]

results = []

for name, model in models:
    # Test setinde tahmin yap
    y_pred = model.predict(X_test)
    
    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MSE": mse,
        "R2": r2
    })

# DataFrame'e dönüştür
results_df = pd.DataFrame(results)

# MSE'yi küçükten büyüğe sıralayalım (daha iyi performans alta değil üste gelsin diye)
results_df = results_df.sort_values(by="MSE")


print(results_df)




