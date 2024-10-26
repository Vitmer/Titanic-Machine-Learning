
# Titanic: Machine Learning from Disaster

This is a machine learning project for the popular Kaggle competition **"Titanic: Machine Learning from Disaster"**. The goal of the project is to predict the survival of passengers aboard the Titanic using various features such as class, age, sex, and more.

## Project Overview

The Titanic dataset is a classic example of a binary classification problem where we predict whether a passenger survived (1) or did not survive (0) based on features such as:

- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Approach

1. **Data Exploration and Preprocessing**:
   - Loaded data from CSV files using `pandas`.
   - Filled missing values in `Age` and `Fare` with their median values and in `Embarked` with the mode.
   - Converted categorical variables like `Sex` and `Embarked` into numerical form using `LabelEncoder`.

2. **Feature Engineering**:
   - Created new features like `FamilySize` (sum of `SibSp` and `Parch` plus one) to capture the family structure of each passenger.
   - Created a binary feature `IsAlone` to indicate if a passenger was traveling alone.

3. **Feature Scaling**:
   - Used `StandardScaler` to normalize features for better model performance.

4. **Model Training**:
   - Split the data into training and validation sets using `train_test_split` with a 95%/5% split.
   - Trained a `RandomForestClassifier` with 100 estimators.

5. **Model Evaluation**:
   - Achieved an accuracy of **88.89%** on the validation set.

## Files in the Repository

- `train.csv`: Training data containing features and labels for passengers.
- `test.csv`: Test data containing features for which predictions need to be made.
- `titanic_submission.csv`: The generated file for submission on Kaggle with the predicted outcomes.

## How to Use the Code

1. **Setup Environment**:
   - Create a virtual environment and install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   
2. **Run the Script**:
   - Use the provided Python script to train the model and generate predictions:
     ```bash
     python titanic.py
     ```

3. **Submission**:
   - After running the script, a `titanic_submission.csv` file will be generated. Upload this file to the Kaggle competition page for scoring.

## Results

With the above approach, the model achieved an accuracy of **88.89%** on the validation set, demonstrating a strong ability to predict passenger survival based on the given features.

## Future Improvements

- Explore other machine learning algorithms like `GradientBoosting` or `XGBoost`.
- Perform hyperparameter tuning using `GridSearchCV` for better results.
- Add more feature engineering techniques to extract additional insights from the data.

## License

This project is provided under the MIT License - see the [LICENSE](LICENSE) file for details.
