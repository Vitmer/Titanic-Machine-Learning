import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. Data exploration and preprocessing
# Fill missing values
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Convert categorical variables into numerical
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])

train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])

# 3. Feature engineering
# Create a new feature 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Create a feature 'IsAlone'
train_data['IsAlone'] = np.where(train_data['FamilySize'] > 1, 0, 1)
test_data['IsAlone'] = np.where(test_data['FamilySize'] > 1, 0, 1)

# Select features for training
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]

# 4. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 5. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

# 6. Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. Model evaluation
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy on validation set: {accuracy * 100:.2f}%')

# 8. Predictions on the test set
test_predictions = rf_model.predict(X_test)

# 9. Prepare the submission file for Kaggle
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)
print("Submission file created.")