Sure! Let’s integrate the **risk prediction model** with **Streamlit** to create an interactive dashboard. The dashboard will allow users to input project metrics and get the predicted risk level, along with visualizations like feature importance and risk distribution.

---

### Streamlit Dashboard Code

Below is the complete code for the Streamlit app:

```python
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Page Title
st.title("Project Risk Prediction Dashboard")
st.write("Predict the risk level of a project based on key metrics.")

# Sample Project Data
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input Project Metrics")
total_tasks = st.sidebar.number_input("Total Tasks", min_value=0, value=50)
completed_tasks = st.sidebar.number_input("Completed Tasks", min_value=0, value=45)
pending_tasks = st.sidebar.number_input("Pending Tasks", min_value=0, value=5)
overdue_tasks = st.sidebar.number_input("Overdue Tasks", min_value=0, value=1)
avg_completion_time = st.sidebar.number_input("Average Completion Time (days)", min_value=0.0, value=3.5)
high_priority_bugs = st.sidebar.number_input("High Priority Bugs", min_value=0, value=2)
resource_availability = st.sidebar.number_input("Resource Availability (%)", min_value=0, max_value=100, value=85)
sprint_velocity = st.sidebar.number_input("Sprint Velocity (tasks/week)", min_value=0, value=12)

# Predict risk for the input project
new_project = pd.DataFrame({
    "total_tasks": [total_tasks],
    "completed_tasks": [completed_tasks],
    "pending_tasks": [pending_tasks],
    "overdue_tasks": [overdue_tasks],
    "avg_completion_time": [avg_completion_time],
    "high_priority_bugs": [high_priority_bugs],
    "resource_availability": [resource_availability],
    "sprint_velocity": [sprint_velocity]
})

predicted_risk = model.predict(new_project)[0]
predicted_risk_level = ["Low", "Medium", "High"][predicted_risk]

# Display the predicted risk level
st.subheader("Predicted Risk Level")
st.write(f"The predicted risk level for the project is: **{predicted_risk_level}**")

# Visualizations
st.subheader("Visualizations")

# 1. Feature Importance
st.write("### Feature Importance")
feature_importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
ax.set_title("Feature Importance for Risk Prediction")
st.pyplot(fig)

# 2. Risk Level Distribution
st.write("### Risk Level Distribution")
y_pred = model.predict(X)
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(y_pred, bins=3, kde=False, color='blue', discrete=True, ax=ax)
ax.set_title("Distribution of Predicted Risk Levels")
ax.set_xlabel("Risk Level")
ax.set_ylabel("Frequency")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Low", "Medium", "High"])
st.pyplot(fig)
```

---

### How to Run the Streamlit App

1. Save the code in a file, e.g., `risk_prediction_dashboard.py`.
2. Install the required libraries:
   ```bash
   pip install streamlit pandas xgboost matplotlib seaborn scikit-learn
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run risk_prediction_dashboard.py
   ```
4. Open the app in your browser (usually at `http://localhost:8501`).

---

### Features of the Streamlit Dashboard

1. **User Input**:
   - Users can input project metrics (e.g., total tasks, completed tasks, overdue tasks) using the sidebar.

2. **Predicted Risk Level**:
   - The app predicts the risk level (`Low`, `Medium`, `High`) based on the input metrics.

3. **Visualizations**:
   - **Feature Importance**: Shows which features contribute most to the risk prediction.
   - **Risk Level Distribution**: Displays the distribution of predicted risk levels across the dataset.

4. **Interactive**:
   - Users can adjust the input metrics and see the predicted risk level update in real-time.

---

### Example Output

1. **Predicted Risk Level**:
   ```
   The predicted risk level for the project is: Medium
   ```

2. **Feature Importance**:
   - A bar chart showing the importance of each feature (e.g., `overdue_tasks`, `resource_availability`).

3. **Risk Level Distribution**:
   - A histogram showing the distribution of predicted risk levels (`Low`, `Medium`, `High`).

---

Let me know if you need further customization or additional features! 🚀
