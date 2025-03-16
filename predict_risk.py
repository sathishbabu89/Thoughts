import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Train the model (same as in your initial code)
def train_model():
    # Sample project data (same as before)
    data = {
        "project_id": ["P001", "P002", "P003", "P004", "P005", "P006", "P007"],
        "sprint_id": ["S01", "S02", "S03", "S04", "S05", "S06", "S07"],
        "total_tasks": [50, 60, 40, 70, 55, 65, 45],
        "completed_tasks": [45, 40, 35, 50, 50, 55, 30],
        "pending_tasks": [5, 20, 5, 20, 5, 10, 15],
        "overdue_tasks": [1, 5, 0, 6, 1, 3, 4],
        "avg_completion_time": [3.5, 6.0, 2.8, 7.5, 3.2, 4.5, 6.5],
        "high_priority_bugs": [2, 5, 1, 6, 2, 3, 4],
        "resource_availability": [85, 60, 90, 55, 80, 75, 60],
        "sprint_velocity": [12, 8, 14, 7, 11, 9, 6],
        "risk_level": ["Low", "High", "Low", "High", "Medium", "Medium", "High"]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Encode the target variable (risk_level)
    df["risk_level"] = df["risk_level"].map({"Low": 0, "Medium": 1, "High": 2})

    # Features and target variable
    X = df.drop(["project_id", "sprint_id", "risk_level"], axis=1)  # Features
    y = df["risk_level"]  # Target variable

    # Initialize and train the XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X, y)

    return model

# Function to process uploaded file and predict risk
def risk_prediction(uploaded_file):
    # Load the trained model
    model = train_model()

    # Read the uploaded Excel file
    df_input = pd.read_excel(uploaded_file)

    # Check if required columns are present
    required_columns = [
        "total_tasks", "completed_tasks", "pending_tasks", "overdue_tasks", 
        "avg_completion_time", "high_priority_bugs", "resource_availability", "sprint_velocity"
    ]
    
    if not all(col in df_input.columns for col in required_columns):
        st.error("The uploaded file must contain the required columns: 'total_tasks', 'completed_tasks', etc.")
        return

    # Predict the risk level for each row in the uploaded file
    X_new = df_input[required_columns]
    predicted_risk = model.predict(X_new)
    
    # Map predictions to risk levels
    risk_levels = ["Low", "Medium", "High"]
    df_input["predicted_risk_level"] = [risk_levels[r] for r in predicted_risk]

    # Display the results
    st.subheader("Predicted Risk Levels for Projects")
    st.write(df_input[["project_id", "predicted_risk_level"]])

    # Risk level metrics (for illustration purposes)
    risk_metrics = {
        "Low": {"total_tasks": 50, "completed_tasks": 45, "pending_tasks": 5},
        "Medium": {"total_tasks": 60, "completed_tasks": 40, "pending_tasks": 20},
        "High": {"total_tasks": 70, "completed_tasks": 30, "pending_tasks": 40},
    }

    st.subheader("Risk Level Metrics")
    st.write(risk_metrics)
