import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy import stats
from yellowbrick.classifier import ConfusionMatrix
from sklearn.compose import ColumnTransformer


# Read the data
data = pd.read_csv("breast-cancer.csv")

# Separating features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Handling outliers using Z-score
z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
outlier_rows = np.where(z_scores > 3)[0]
X = X.drop(outlier_rows)
y = y.drop(outlier_rows)

# Preprocess categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=[np.number]).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = preprocessor.fit_transform(X)

# Oversampling using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the MLP Classifier
model = MLPClassifier(max_iter=1000, verbose=True, tol=1e-14, solver='adam', activation='relu', hidden_layer_sizes=(9))
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the confusion matrix
cm = ConfusionMatrix(model)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (10, 5), (20, 10), (30, 15)],
    'activation': ['relu', 'tanh', 'logistic'],
}

mlp = MLPClassifier(max_iter=1000, random_state=42)

grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best parameters:", best_params)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the best model:", accuracy)
