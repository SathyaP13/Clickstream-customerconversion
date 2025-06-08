# Clickstream-customerconversion
- This project features a Streamlit web application that leverages clickstream data to significantly enhance customer engagement and drive sales for e-commerce businesses.
- It tackles key business challenges through predictive modeling and customer segmentation.

## 🚀 Project Overview
This application transforms raw Browse behavior into actionable insights, enabling data-driven decisions that ultimately boost conversions, increase revenue, and improve customer satisfaction.

## ✨ Core Functionalities
- The application integrates three primary machine learning capabilities:
  * Conversion Prediction: Forecasts customer purchase likelihood.
  * Revenue Projection: Estimates potential customer spending, aiding in optimizing pricing and promotional strategies.
  * Customer Segmentation: Groups users based on online behavior, facilitating personalized product recommendations.

## 🛠️ Technical Stack
- Programming: Python
- Data Handling: Pandas, NumPy
- Visualizations: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost, TensorFlow
- Web Framework: Streamlit
- Methodologies: Data Preparation, Feature Engineering, Machine Learning Pipelines

## 🎯 Development Approach
- The project follows a comprehensive data science workflow:
  * Data Preparation:
      - Missing Values: Imputing numerical data with central tendencies and categorical data with modes.
      - Feature Transformation: Encoding categorical attributes and scaling numerical features.
  * Exploratory Data Analysis (EDA):
      - Visualizing data distributions and inter-feature relationships.
      - Analyzing session metrics like duration, page views, and bounce rates.
  * Identifying feature correlations:
      - Extracting temporal features (e.g., hour, day of week).
  * Feature Engineering:
      - Deriving session-specific metrics such as length, click count, and time spent per product category.
      - Tracing click sequences to uncover Browse patterns.
      - Calculating behavioral indicators like exit rates and revisit frequency.
      - Addressing Class Imbalance (Classification)
  * Analyzing target label distribution:
      - Employing techniques such as SMOTE (oversampling), undersampling, or class weight adjustments.
## Model Development
### Supervised Learning:
  - Classification: Logistic Regression, Decision Trees, Random Forest, XGBoost, Neural Networks.
  - Regression: Linear Regression, Ridge, Lasso, Gradient Boosting Regressors.
### Unsupervised Learning:
  - Clustering: K-means, DBSCAN, Hierarchical Clustering.
  - Utilizing Scikit-learn Pipelines for streamlined data processing, scaling, model training, hyperparameter tuning, and evaluation.
## Model Evaluation
  - Classification: Assessing with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Regression: Evaluating using MAE, MSE, RMSE, and R-squared.
  - Clustering: Measuring effectiveness with Silhouette Score, Davies-Bouldin Index, and Within-Cluster Sum of Squares.
## Streamlit Application
  - Developing an interactive web interface supporting CSV uploads or manual input.
  - Displaying real-time conversion predictions and revenue estimations.
  - Visualizing customer segments and key insights through various charts.

## 🛠️ Prerequisites
  - Python 3.8+

## 🚀 Getting Started
  - Install the required packages:
    - pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib seaborn streamlit
## 🚀 Running the Project
  - Place the Projectfinal.ipynb and Clickstreamconversion.py(streamlit app) in a project folder.
  - Run the Projectfinal.ipynb python notebook.
  - All the required data for running the streamlit app will get stored in the project's directory.
  - Then, run the Clickstreamconversion.py app and a browser with instructions will open that will enable the user to hover over various data.
## 🤝 Support & Contribution
- Contributions to this project are Welcomed!!! Feel free to fork this repository, open issues for bugs or feature requests, and submit pull requests.
