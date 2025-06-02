# Stroke Prediction Project

This project uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) from Kaggle to build a machine learning model that predicts the likelihood of a patient having a stroke based on various health and demographic features.

## Dataset Information

The dataset includes the following features:

- `id`: Unique identifier
- `gender`: Gender of the patient
- `age`: Age of the patient
- `hypertension`: 0 = No, 1 = Yes
- `heart_disease`: 0 = No, 1 = Yes
- `ever_married`: Whether the patient has ever been married
- `work_type`: Type of work (e.g., Private, Self-employed, Govt job)
- `Residence_type`: Urban or Rural
- `avg_glucose_level`: Average glucose level in blood
- `bmi`: Body Mass Index
- `smoking_status`: Smoking habits (e.g., formerly smoked, never smoked, smokes)
- `stroke`: Target variable (0 = No stroke, 1 = Stroke)

## Project Goals

- Clean and preprocess the data
- Handle missing values and encode categorical features
- Explore feature importance and visualizations
- Train and evaluate multiple classification models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - XGBoost (optional)
- Evaluate models using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Compare results with baseline models
- Use techniques such as feature engineering, hyperparameter tuning, and ensemble methods to improve model performance


## Data Analysis and Visualization

- Use tables to display summary statistics and data distributions
- Visualize relationships and correlations between features using plots

## Usage of Tables and Plots

1. **Summary Statistics Table**: Display mean, median, standard deviation, etc., for numerical features using a library like `pandas`.

2. **Correlation Matrix**: Use a heatmap to visualize correlations between features using `seaborn`.

3. **Feature Distribution**: Plot histograms or box plots to understand the distribution of each feature.

4. **Pair Plot**: Visualize pairwise relationships between features using `seaborn`'s `pairplot`.

5. **Visualization Tools**: Utilize libraries like `matplotlib` and `seaborn` for creating plots.

