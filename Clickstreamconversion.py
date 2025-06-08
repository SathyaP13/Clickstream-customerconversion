import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Customer Behavior Analysis & Prediction")

# --- Predefined Options from your Mappings ---
COUNTRY_OPTIONS = [
    "Australia", "Austria", "Belgium", "British Virgin Islands", "Cayman Islands",
    "Christmas Island", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
    "unidentified", "Faroe Islands", "Finland", "France", "Germany", "Greece",
    "Hungary", "Iceland", "India", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Mexico", "Netherlands", "Norway", "Poland", "Portugal",
    "Romania", "Russia", "San Marino", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Ukraine", "United Arab Emirates", "United Kingdom",
    "USA", "biz", "com", "int", "net", "org" 
]

COLOUR_OPTIONS = [
    "beige", "black", "blue", "brown", "burgundy", "gray", "green",
    "navy blue", "of many colors", "olive", "pink", "red", "violet", "white"
]

LOCATION_OPTIONS = [
    "top left", "top middle", "top right", "bottom left", "bottom middle", "bottom right"
]

PAGE1_MAIN_CATEGORY_OPTIONS = [
    "trousers", "skirts", "blouses", "sale"
]

MODEL_PHOTOGRAPHY_OPTIONS = ["en face", "profile"]


# --- Load Models and Preprocessor ---
@st.cache_resource
def load_assets():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        clf_model = joblib.load('best_xgb_clf.pkl')
        reg_model = joblib.load('best_xgb_reg.pkl')
        kmeans_model = joblib.load('kmeans_final.pkl')
        return preprocessor, clf_model, reg_model, kmeans_model
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure train_model.py has been run successfully and all .pkl files are in the same directory.")
        st.stop() # Stop execution if models are not found
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop() # Stop execution if models cannot be loaded

preprocessor, best_xgb_clf, best_xgb_reg, kmeans_final = load_assets()

# --- Helper function to get preprocessor's column names ---
def get_preprocessor_features(preprocessor):
    # This assumes the order of transformers is 'num', 'cat' as set up in Projectfinal.ipynb
    # and that each transformer's get_feature_names_out() method works.
    try:
        numerical_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        return list(numerical_features) + list(categorical_features)
    except Exception as e:
        st.error(f"Error getting feature names from preprocessor: {e}. Ensure preprocessor was fitted correctly.")
        # Fallback for debugging, might not be accurate if preprocessor is complex
        return preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else []


# --- Feature Engineering and Preprocessing Function ---
def preprocess_input(input_df_raw, preprocessor):
    # Ensure all expected columns are present, fill with defaults or 'Others' if missing for robustness
    expected_cols = ['session_id', 'order', 'page', 'price', 'price_2',
                     'year', 'month', 'day', 'country', 'page1_main_category',
                     'colour', 'location', 'model_photography']
    for col in expected_cols:
        if col not in input_df_raw.columns:
            if col in ['order', 'page', 'year', 'month', 'day', 'price', 'price_2']:
                input_df_raw[col] = 0 # Numerical default
            else:
                input_df_raw[col] = 'Others' # Categorical default

    processed_df = input_df_raw.copy()

    # Feature Engineering (replicate logic from Projectfinal.ipynb based on problem description)
    session_agg = processed_df.groupby('session_id').agg(
        session_length=('order', 'count'),
        max_order=('order', 'max'),
        min_order=('order', 'min'),
        avg_price_per_session=('price', 'mean'),
        max_price_per_session=('price', 'max'),
        min_price_per_session=('price', 'min'),
        num_unique_page1_categories=('page1_main_category', lambda x: x.nunique()),
        num_unique_countries=('country', lambda x: x.nunique()),
        max_page_visited=('page', 'max'),
        min_page_visited=('page', 'min')
    ).reset_index()

    session_level_features_modes = processed_df.groupby('session_id').agg(
        year=('year', 'first'),
        month=('month', lambda x: x.mode()[0] if not x.mode().empty else 1),
        day=('day', lambda x: x.mode()[0] if not x.mode().empty else 1),
        country=('country', lambda x: x.mode()[0] if not x.mode().empty else 'Others'),
        page1_main_category=('page1_main_category', lambda x: x.mode()[0] if not x.mode().empty else 'Others'),
        colour=('colour', lambda x: x.mode()[0] if not x.mode().empty else 'Others'),
        location=('location', lambda x: x.mode()[0] if not x.mode().empty else 'Others'),
        model_photography=('model_photography', lambda x: x.mode()[0] if not x.mode().empty else 'Others'), 
        price_avg_click=('price', 'mean'), 
        price2_mode=('price_2', lambda x: x.mode()[0] if not x.mode().empty else 0.0),
    ).reset_index()

    session_level_features = pd.merge(session_level_features_modes, session_agg, on='session_id', how='left')

    session_level_features['date'] = pd.to_datetime(session_level_features[['year', 'month', 'day']], errors='coerce')
    session_level_features['day_of_week'] = session_level_features['date'].dt.dayofweek
    session_level_features['is_weekend'] = session_level_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Handle NaT values from date conversion
    session_level_features['day_of_week'] = session_level_features['day_of_week'].fillna(-1).astype(int)
    session_level_features['is_weekend'] = session_level_features['is_weekend'].fillna(-1).astype(int)

    session_level_features.drop(columns=['year', 'month', 'day', 'date'], inplace=True)

    # Store session_ids before dropping for prediction output
    session_ids = session_level_features['session_id'].copy()
    features_for_transform = session_level_features.drop(columns=['session_id'])

    try:
        transformed_features = preprocessor.transform(features_for_transform)

        # Get feature names from the preprocessor after transformation
        all_final_cols = get_preprocessor_features(preprocessor)

        # CRITICAL FIX for KeyError: Set index of transformed_df to session_ids values
        # This makes input_data consistently indexed by actual session_ids for .loc operations
        transformed_df = pd.DataFrame(transformed_features.toarray(), columns=all_final_cols, index=session_ids.values)

    except Exception as e:
        st.error(f"Error during preprocessing transform: {e}")
        st.error("This might be due to unexpected columns, missing columns, or different data types in your input file compared to the training data.")
        st.stop() # Stop if preprocessing fails

    # CRITICAL FIX for Ambiguous Error: Ensure session_level_features does not have a named index
    session_level_features.index.name = None

    return transformed_df, session_ids, session_level_features # Return session_level_features for info/display

# --- Main Streamlit App ---
st.title("Customer Purchase Prediction & Revenue Forecasting")

st.markdown("""
This application predicts whether a customer session will lead to a purchase and, if so, estimates the revenue generated.
It also provides insights into customer segmentation.
""")

# --- Sidebar for Input Method and Model Details ---
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ("Upload CSV", "Manual Input for a Single Session"))

st.sidebar.header("Model Information")
st.sidebar.info("""
**Classification Model:** XGBoost Classifier (Predicts 'Is_Purchase')
**Regression Model:** XGBoost Regressor (Predicts 'Revenue_Generated' for purchases)
**Clustering Model:** K-Means (For Customer Segmentation)

Models were trained on 'train_data.csv' and 'test_data.csv' using `train_model.py`.
""")

input_data = None
session_ids = None
session_level_features_display = None # This will hold the feature-engineered but non-preprocessed data + session_id

if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your session data CSV", type="csv")
    if uploaded_file is not None:
        st.write("Processing uploaded data...")
        try:
            input_df_raw = pd.read_csv(uploaded_file)
            st.subheader("Raw Input Data Head:")
            st.dataframe(input_df_raw.head())
            input_data, session_ids, session_level_features_display = preprocess_input(input_df_raw, preprocessor)
            st.write(f"Processed features shape for prediction: {input_data.shape}")
            st.subheader("Preprocessed Features (Head):")
            st.dataframe(input_data.head())
        except Exception as e:
            st.error(f"An error occurred during CSV upload or preprocessing: {e}")
            st.info("Please ensure your CSV is correctly formatted and contains the expected columns.")
    else:
        st.info("Please upload a CSV file to proceed.")

elif input_method == "Manual Input for a Single Session":
    st.subheader("Manual Input for a Single Session")
    with st.form("manual_input_form"):
        # Basic session details
        col1, col2, col3 = st.columns(3)
        with col1:
            session_id_manual = st.number_input("Session ID", value=100000, min_value=1)
            order_manual = st.number_input("Order (e.g., 1 for first item)", value=1, min_value=1)
            page_manual = st.number_input("Page Visited (1-5, 5 for purchase)", value=1, min_value=1, max_value=5)
            price_manual = st.number_input("Item Price", value=10.0, min_value=0.0)
        with col2:
            price2_manual = st.number_input("Price 2 (from dataset)", value=0.0, min_value=0.0)
            year_manual = st.number_input("Year", value=2008, min_value=2008, max_value=2008) # Year fixed to 2008 as per dataset description
            month_manual = st.number_input("Month", value=6, min_value=4, max_value=8) # Month from April(4) to August(8)
            day_manual = st.number_input("Day", value=6, min_value=1, max_value=31)
        with col3:
            # Using selectbox for country, colour, location, page1_main_category, model_photography
            country_manual = st.selectbox("Country", options=COUNTRY_OPTIONS, index=COUNTRY_OPTIONS.index("Germany") if "Germany" in COUNTRY_OPTIONS else 0)
            page1_main_category_manual = st.selectbox("Main Category (Page 1)", options=PAGE1_MAIN_CATEGORY_OPTIONS, index=PAGE1_MAIN_CATEGORY_OPTIONS.index("blouses") if "blouses" in PAGE1_MAIN_CATEGORY_OPTIONS else 0)
            colour_manual = st.selectbox("Colour", options=COLOUR_OPTIONS, index=COLOUR_OPTIONS.index("black") if "black" in COLOUR_OPTIONS else 0)
            location_manual = st.selectbox("Location", options=LOCATION_OPTIONS, index=LOCATION_OPTIONS.index("top left") if "top left" in LOCATION_OPTIONS else 0)
            model_photography_manual = st.selectbox("Model Photography", options=MODEL_PHOTOGRAPHY_OPTIONS, index=MODEL_PHOTOGRAPHY_OPTIONS.index("profile") if "profile" in MODEL_PHOTOGRAPHY_OPTIONS else 0)


        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            manual_input_df = pd.DataFrame([{
                'session_id': session_id_manual,
                'order': order_manual,
                'page': page_manual,
                'price': price_manual,
                'price_2': price2_manual,
                'year': year_manual,
                'month': month_manual,
                'day': day_manual,
                'country': country_manual,
                'page1_main_category': page1_main_category_manual,
                'colour': colour_manual,
                'location': location_manual,
                'model_photography': model_photography_manual
            }])
            st.subheader("Raw Manual Input:")
            st.dataframe(manual_input_df)
            input_data, session_ids, session_level_features_display = preprocess_input(manual_input_df, preprocessor)
            st.write(f"Processed features shape for prediction: {input_data.shape}")
            st.subheader("Preprocessed Features (Manual Input):")
            st.dataframe(input_data)
        else:
            st.info("Enter values and click 'Get Prediction' to analyze a single session.")

# --- Perform Predictions and Display Results ---
if input_data is not None and not input_data.empty:
    st.markdown("---")
    st.subheader("Prediction Results")

    # Initialize prediction_df with session_ids as its index
    prediction_df = pd.DataFrame({
        'session_id': session_ids,
        'Is_Purchase_Predicted': 0, # Default to 0
        'Purchase_Probability': 0.0, # Default to 0.0
        'Predicted_Revenue': 0.0 # Initialize Predicted_Revenue
    }).set_index('session_id')


    # Make Classification Predictions
    st.markdown("#### Purchase Conversion Prediction")
    if best_xgb_clf:
        is_purchase_predictions = best_xgb_clf.predict(input_data)
        purchase_probabilities = best_xgb_clf.predict_proba(input_data)[:, 1]

        prediction_df['Is_Purchase_Predicted'] = is_purchase_predictions
        prediction_df['Purchase_Probability'] = purchase_probabilities

        if input_method == "Manual Input for a Single Session":
            st.write(f"**Predicted Purchase:** {'Yes' if is_purchase_predictions[0] == 1 else 'No'}")
            st.write(f"**Purchase Probability:** {purchase_probabilities[0]:.2f}")
        else:
            st.write(f"Number of sessions predicted as purchase: {np.sum(is_purchase_predictions)}")
            st.write(f"Number of sessions predicted as non-purchase: {len(is_purchase_predictions) - np.sum(is_purchase_predictions)}")

    else:
        st.warning("Classification model not loaded or trained successfully. Cannot predict purchases.")


    # Make Regression Predictions (only for predicted purchases)
    st.markdown("#### Revenue Forecasting")
    if best_xgb_reg:
        # Filter input_data for sessions predicted as purchase
        # input_data is correctly indexed by session_id, and prediction_df is also indexed by session_id
        predicted_purchase_mask = (prediction_df['Is_Purchase_Predicted'] == 1)
        predicted_purchase_session_ids_for_reg = predicted_purchase_mask.index[predicted_purchase_mask] # Get the actual session IDs

        predicted_purchase_sessions_df_for_reg = input_data.loc[predicted_purchase_session_ids_for_reg]


        if not predicted_purchase_sessions_df_for_reg.empty:
            revenue_predictions_for_purchases = best_xgb_reg.predict(predicted_purchase_sessions_df_for_reg)
            revenue_predictions_for_purchases[revenue_predictions_for_purchases < 0] = 0 # Ensure no negative revenue

            # Update 'Predicted_Revenue' column in prediction_df using .loc
            prediction_df.loc[predicted_purchase_session_ids_for_reg, 'Predicted_Revenue'] = revenue_predictions_for_purchases.round(2)

            if input_method == "Manual Input for a Single Session":
                st.write(f"**Predicted Revenue:** ${prediction_df['Predicted_Revenue'].iloc[0]:,.2f}")
            else:
                st.write(f"Total predicted revenue for purchase sessions: ${prediction_df['Predicted_Revenue'].sum():,.2f}")
        else:
            st.info("No sessions predicted as purchase, so no revenue forecast generated.")
    else:
        st.warning("Regression model not loaded or trained successfully. Cannot forecast revenue.")

    st.subheader("Summary of Predictions")
    # Reset index for display and download to show session_id as a column
    st.dataframe(prediction_df.reset_index().head())

    if input_method == "Upload CSV":
        csv_output = prediction_df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_output,
            file_name="session_predictions_app.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # --- Customer Segmentation (Clustering) ---
    st.subheader("Customer Segmentation")
    if kmeans_final and input_data.shape[0] > 0:
        try:
            cluster_labels = kmeans_final.predict(input_data)

            # session_level_features_display already has session_id as a column and its index name is None
            session_level_features_display['Cluster'] = cluster_labels

            # Merge Predicted_Revenue back into session_level_features_display for cluster analysis
            # Ensure prediction_df has session_id as a column for merging (reset_index does this)
            session_level_features_display_with_revenue = pd.merge(
                session_level_features_display,
                prediction_df.reset_index()[['session_id', 'Predicted_Revenue', 'Is_Purchase_Predicted']], # Include Is_Purchase_Predicted for more insights
                on='session_id',
                how='left'
            )
            session_level_features_display_with_revenue['Predicted_Revenue'] = session_level_features_display_with_revenue['Predicted_Revenue'].fillna(0)
            session_level_features_display_with_revenue['Is_Purchase_Predicted'] = session_level_features_display_with_revenue['Is_Purchase_Predicted'].fillna(0).astype(int) # Fill and cast


            st.write("Distribution of Sessions Across Clusters:")
            fig_cluster_dist = plt.figure(figsize=(8, 5)) # NEW FIGURE for every plot
            cluster_counts = session_level_features_display_with_revenue['Cluster'].value_counts().sort_index()
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
            plt.xlabel('Cluster')
            plt.ylabel('Number of Sessions')
            plt.title('Distribution of Sessions Across Clusters')
            st.pyplot(fig_cluster_dist)
            plt.close(fig_cluster_dist) # Close figure to free memory

            st.write("Customer Segmentation Insights (Head of data with clusters):")
            st.dataframe(session_level_features_display_with_revenue.head())

            st.markdown("##### Average Metrics per Cluster:")
            cluster_summary_with_revenue = session_level_features_display_with_revenue.groupby('Cluster').agg(
                Avg_Session_Length=('session_length', 'mean'),
                Avg_Price_Per_Session=('avg_price_per_session', 'mean'),
                Total_Predicted_Revenue=('Predicted_Revenue', 'sum'),
                Num_Sessions=('session_id', 'count'),
                Purchase_Conversion_Rate=('Is_Purchase_Predicted', 'mean') 
            ).round(2)
            st.dataframe(cluster_summary_with_revenue)


            st.markdown("##### Visualizations per Cluster (Example: Session Length & Revenue)")
            fig_session_length = plt.figure(figsize=(10, 6)) # NEW FIGURE
            sns.boxplot(x='Cluster', y='session_length', data=session_level_features_display_with_revenue)
            plt.title('Session Length Distribution Across Clusters')
            plt.xlabel('Cluster')
            plt.ylabel('Session Length')
            st.pyplot(fig_session_length)
            plt.close(fig_session_length) # Close figure

            fig_revenue_cluster = plt.figure(figsize=(10, 6)) # NEW FIGURE
            sns.barplot(x='Cluster', y='Predicted_Revenue', data=session_level_features_display_with_revenue.groupby('Cluster')['Predicted_Revenue'].sum().reset_index())
            plt.title('Total Predicted Revenue per Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Total Predicted Revenue ($)')
            st.pyplot(fig_revenue_cluster)
            plt.close(fig_revenue_cluster) # Close figure


        except Exception as e:
            st.warning(f"Error during clustering: {e}. Ensure the clustering model was trained successfully and input data is compatible.")
    else:
        st.info("Clustering model not loaded or trained successfully, or no data to cluster.")

    st.markdown("---")
    # --- General Data Visualizations ---
    st.subheader("Input Data Visualizations")
    if session_level_features_display is not None and not session_level_features_display.empty:
        st.write("Explore distributions of key features from your input data.")

        # Numerical features histogram
        numerical_cols_for_viz = ['session_length', 'price_avg_click', 'avg_price_per_session', 'max_page_visited']
        selected_numerical_col = st.selectbox("Select Numerical Feature for Histogram:", numerical_cols_for_viz, key='num_hist_select')
        if selected_numerical_col in session_level_features_display.columns:
            fig_hist = plt.figure(figsize=(10, 6)) # NEW FIGURE
            sns.histplot(session_level_features_display[selected_numerical_col], kde=True, bins=20)
            plt.title(f'Distribution of {selected_numerical_col}')
            plt.xlabel(selected_numerical_col)
            plt.ylabel('Frequency')
            st.pyplot(fig_hist)
            plt.close(fig_hist) # Close figure
        else:
            st.warning(f"Column '{selected_numerical_col}' not found in processed input data for visualization.")


        # Categorical features bar/pie chart
        categorical_cols_for_viz = ['country', 'page1_main_category', 'colour', 'location', 'model_photography', 'day_of_week', 'is_weekend']
        selected_categorical_col = st.selectbox("Select Categorical Feature for Bar Chart:", categorical_cols_for_viz, key='cat_bar_select')
        if selected_categorical_col in session_level_features_display.columns:
            fig_bar = plt.figure(figsize=(12, 7)) # NEW FIGURE
            sns.countplot(y=selected_categorical_col, data=session_level_features_display, order=session_level_features_display[selected_categorical_col].value_counts().index)
            plt.title(f'Count of Sessions by {selected_categorical_col}')
            plt.xlabel('Number of Sessions')
            plt.ylabel(selected_categorical_col)
            st.pyplot(fig_bar)
            plt.close(fig_bar) # Close figure
        else:
            st.warning(f"Column '{selected_categorical_col}' not found in processed input data for visualization.")

    else:
        st.info("Upload data or provide manual input to see visualizations.")