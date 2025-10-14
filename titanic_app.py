import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------
# 1️⃣ Load and prepare Titanic data
# ------------------------------
df = sns.load_dataset('titanic')

# Handle missing data (fixed warnings by assigning back)
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Features and target
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set for accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------
# 2️⃣ Streamlit App
# ------------------------------
st.title("🚢 Titanic Survival Prediction App")
st.subheader(f"Model Accuracy: {accuracy*100:.2f}%")

# ------------------------------
# 3️⃣ Sidebar for user inputs
# ------------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3])
sex_input = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Ticket Fare", 0.0, 512.0, 32.0)
embarked_input = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs to numbers
sex = 0 if sex_input == "Male" else 1
embarked = {"C": 0, "Q": 1, "S": 2}[embarked_input]

# Predict button
if st.sidebar.button("Predict Survival"):
    # Convert input to DataFrame (fixes warning)
    input_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=features)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    # Show survival prediction
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(result)
    st.write(f"Prediction Confidence: {probability*100:.2f}%")

    # ------------------------------
    # Feature contributions (why survived / why not)
    # ------------------------------
    contributions = input_df.iloc[0] * model.coef_[0]
    contrib_df = pd.DataFrame({
        'Feature': features,
        'Value': input_df.iloc[0],
        'Contribution': contributions
    }).sort_values(by='Contribution', key=abs, ascending=False)

    st.subheader("Why did the model predict this?")
    for i, row in contrib_df.iterrows():
        sign = "increased" if row['Contribution'] > 0 else "decreased"
        st.write(f"- {row['Feature']} = {row['Value']} → {sign} chance of survival")

    # ------------------------------
    # Show real answer using approximate matching
    # ------------------------------
    tolerance_age = 1      # years
    tolerance_fare = 5.0   # dollars

    match = df[
        (df['pclass'] == pclass) &
        (df['sex'] == sex) &
        (abs(df['age'] - age) <= tolerance_age) &
        (df['sibsp'] == sibsp) &
        (df['parch'] == parch) &
        (abs(df['fare'] - fare) <= tolerance_fare) &
        (df['embarked'] == embarked)
    ]

    if not match.empty:
        real_survival = match['survived'].values[0]
        real_result = "🟢 Survived" if real_survival == 1 else "🔴 Did Not Survive"
        st.subheader(f"Closest Real Answer: {real_result}")
    else:
        st.subheader("No close match found in dataset (no real answer available)")

# ------------------------------
# 4️⃣ Charts
# ------------------------------
st.header("Data Visualization")

# Survival count by gender
st.subheader("Survival Count by Gender")
fig1, ax1 = plt.subplots()
sns.countplot(x='survived', hue='sex', data=df, ax=ax1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Did Not Survive", "Survived"])
st.pyplot(fig1)

# Survival rate by passenger class
st.subheader("Survival Rate by Passenger Class")
fig2, ax2 = plt.subplots()
sns.barplot(x='pclass', y='survived', data=df, ax=ax2)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
st.pyplot(fig2)
