import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.graph_objects as go
import joblib  # For saving and loading models
import os  # To check if model files exist
import resource_op as ro

def predict():
    import streamlit as st
    import pandas as pd
    import joblib  # For loading models

    # Load the pre-trained models (assumes you saved them previously as shown)
    cost_model = joblib.load("cost_model.pkl")  # Load your cost prediction model
    resource_model = joblib.load("resource_model.pkl")  # Load your resource scheduling model

    # Title of the app
    st.title("Coop Store Service Cost and Resource Prediction")

    # User input form for store details
    st.subheader("Enter Store Details")

    # Input fields for store characteristics
    store_size = st.number_input("Store Size (SQ/F)", min_value=500, max_value=5000, value=3000, step=100)
    productivity = st.number_input("Productivity (SQ/F per hour)", min_value=50, max_value=500, value=200, step=10)
    demand_score = st.slider("Demand Score", 1, 10, 5)
    priority_level = st.selectbox("Priority Level", [1, 2, 3],
                                  format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High'}[x])

    # Get postcode from user
    postcode = st.text_input("Enter Postcode for Current Location")
    distance_threshold = st.number_input("Maximum Distance for Nearby Stores (in miles)", min_value=1, value=5, step=1)

    # Convert user input to DataFrame for model prediction
    input_data = pd.DataFrame({
        'Store_Size_SQFT': [store_size],
        'Productivity_SQ_F_PerHour': [productivity],
        'Demand_Score': [demand_score],
        'Priority_Level': [priority_level]
    })

    # Predict on button click
    if st.button("Predict Service Cost and Resource Type"):
        # Predict cost and resource type using models
        predicted_cost = cost_model.predict(input_data)[0]
        predicted_resource = resource_model.predict(input_data)[0]
        resource_type = 'Fixed' if predicted_resource == 1 else 'Mobile'

        # Display results
        st.write("### Prediction Results:")
        st.write(f"**Estimated Service Cost per Visit**: £{predicted_cost:.2f}")
        st.write(f"**Recommended Resource Type**: {resource_type}")

    # Optional: If you want to allow users to upload a dataset and apply the model on it
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type="csv")
    if uploaded_file:
        # Read the uploaded CSV
        user_data = pd.read_csv(uploaded_file)
        # Apply the same transformations as the input data and make predictions
        user_data['Priority_Level'] = user_data['Priority_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
        user_data['Predicted_Cost'] = cost_model.predict(
            user_data[["Store_Size_SQFT", "Productivity_SQ_F_PerHour", "Demand_Score", "Priority_Level"]])
        user_data['Predicted_Resource_Type'] = resource_model.predict(
            user_data[["Store_Size_SQFT", "Productivity_SQ_F_PerHour", "Demand_Score", "Priority_Level"]])
        user_data['Predicted_Resource_Type'] = user_data['Predicted_Resource_Type'].map({0: 'Mobile', 1: 'Fixed'})

        st.write("### Batch Prediction Results:")
        st.dataframe(user_data[['Store_Size_SQFT', 'Productivity_SQ_F_PerHour', 'Demand_Score', 'Priority_Level',
                                'Predicted_Cost', 'Predicted_Resource_Type']])


def main():
    st.title("Store Data Analysis App")
    st.write("Analyze and filter Coop store data.")

    # Step 1: Load the data
    file_path = "coop_new_dataset.csv"
    if file_path:
        df = pd.read_csv(file_path, dtype={"Karcher reference": str})
        st.write("Data Loaded:")
        st.dataframe(df)

        # Encode 'Priority_Level' for model compatibility
        df['Priority_Level'] = df['Priority_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})

        # Define cost prediction target
        df['Total_Estimated_Cost'] = df['Estimated_Service_Time_Hours'] * df['Cost_Per_Hour_Labour_Only']

        # Encode Resource_Type for classification
        df['Resource_Type_Encoded'] = df['Resource_Type'].map({'Mobile': 0, 'Fixed': 1})

        # Define features for both models
        features = ['Store_Size_SQFT', 'Productivity_SQ_F_PerHour', 'Demand_Score', 'Priority_Level']

        # Cost Prediction Target (Regression)
        y_cost = df['Total_Estimated_Cost']

        # Resource Scheduling Target (Classification)
        y_resource = df['Resource_Type_Encoded']

        # Split data for cost prediction
        X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(df[features], y_cost, test_size=0.2, random_state=42)

        # Split data for resource scheduling
        X_train_resource, X_test_resource, y_train_resource, y_test_resource = train_test_split(df[features], y_resource, test_size=0.2, random_state=42)

        # Check if saved models exist
        if os.path.exists("cost_model.pkl") and os.path.exists("resource_model.pkl"):
            # Load the saved models
            cost_model = joblib.load("cost_model.pkl")
            resource_model = joblib.load("resource_model.pkl")
            st.write("Loaded saved models.")
        else:
            # Train models if not already saved
            cost_model = RandomForestRegressor(random_state=42)
            cost_model.fit(X_train_cost, y_train_cost)
            joblib.dump(cost_model, "cost_model.pkl")  # Save cost model

            resource_model = RandomForestClassifier(random_state=42)
            resource_model.fit(X_train_resource, y_train_resource)
            joblib.dump(resource_model, "resource_model.pkl")  # Save resource model

            st.write("Trained and saved models.")

        # Predict cost on test data
        predicted_cost = cost_model.predict(X_test_cost)

        # Predict resource type on test data
        predicted_resource = resource_model.predict(X_test_resource)

        # Evaluate cost prediction model
        mse_cost = mean_squared_error(y_test_cost, predicted_cost)
        rmse_cost = np.sqrt(mse_cost)
        st.write(f"Root Mean Squared Error score for Cost Prediction: {rmse_cost}, means that, on average, our predicted costs are about £{rmse_cost} away from the actual costs. This gives us a good sense of how accurate our estimates are and helps us make informed decisions based on this data.")

        # Evaluate resource scheduling model
        accuracy_resource = accuracy_score(y_test_resource, predicted_resource)
        st.write(f"Accuracy for Resource Scheduling Prediction: {accuracy_resource}")

        # Sample of predictions for verification
        sample_predictions = pd.DataFrame({
            "Actual Cost": y_test_cost.reset_index(drop=True)[:10],
            "Predicted Cost": predicted_cost[:10],
            "Actual Resource Type": y_test_resource.reset_index(drop=True)[:10].map({0: 'Mobile', 1: 'Fixed'}),
            "Predicted Resource Type": pd.Series(predicted_resource[:10]).map({0: 'Mobile', 1: 'Fixed'})
        })

        st.dataframe(sample_predictions)

        # Create Plotly line chart
        fig = go.Figure()

        # Add Actual Cost line
        fig.add_trace(go.Scatter(x=sample_predictions.index, y=sample_predictions['Actual Cost'],
                                 mode='lines+markers', name='Actual Cost'))

        # Add Predicted Cost line
        fig.add_trace(go.Scatter(x=sample_predictions.index, y=sample_predictions['Predicted Cost'],
                                 mode='lines+markers', name='Predicted Cost'))

        # Update layout
        fig.update_layout(title="Actual vs Predicted Cost",
                          xaxis_title="Test Sample Index",
                          yaxis_title="Cost",
                          template="plotly_white")

        # Display Plotly chart with a unique key
        st.plotly_chart(fig, key="cost_prediction_chart")

        ro.predict1()


if __name__ == "__main__":
    main()
