import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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

st.markdown("""
---
### 🧭 About This App

In April 1912, the **RMS Titanic**, one of the largest ships of its time, struck an iceberg and sank in the North Atlantic Ocean.  
Out of **2,200+ passengers**, only about **38% survived**.

This app uses **Machine Learning (Logistic Regression)** to predict whether a passenger *might have survived* the Titanic disaster based on their:
- 🛳️ Passenger class  
- 👩 Gender  
- 🎂 Age  
- 👨‍👩‍👧 Family members aboard (siblings/spouses and parents/children)  
- 💰 Ticket fare  
- ⚓ Port of embarkation  

The goal is to explore **how these factors influenced survival chances** — and to understand patterns like why:

**“👩 Women and 👶 children had better survival chances than 👨 men.”**

Try adjusting the details in the sidebar to see **how survival probability changes!**
---
""")

# 🧩 INSERT HERE: Step-by-step usage guide
with st.expander("🪜 Step-by-Step Guide"):
    st.write("""
    1. Choose passenger details from the sidebar.  
    2. Click **Predict Survival** to see if they would survive.  
    3. Explore *why* the model predicted that result.  
    4. Scroll down to learn how the data looks and what affects survival.
    """)

# 🧩 INSERT HERE: Explain the model
with st.expander("📘 How this model works"):
    st.write("""
    This app uses **Logistic Regression**, a simple machine learning algorithm
    that predicts whether someone **survived or not** based on features like
    age, gender, ticket class, and fare.

    - **Input features**: Passenger details  
    - **Target**: Whether they survived (1) or not (0)  
    - **Output**: Probability of survival
    """)

# ------------------------------
# 3️⃣ Sidebar for user inputs
# ------------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3], key="pclass")
sex_input = st.sidebar.selectbox("Sex", ["Male", "Female"], key="sex")
age = st.sidebar.slider("Age", min_value=0, max_value=100, key="age")
sibsp = st.sidebar.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, key="sibsp")
parch = st.sidebar.number_input("Parents/Children aboard", min_value=0, max_value=10, key="parch")
fare = st.sidebar.slider("Fare", min_value=0.0, max_value=600.0, key="fare")
embarked_input = st.sidebar.selectbox("Port of Embarkation", ["C = Cherbourg", "Q = Queenstown", "S = Southampton"])

# 🧩 INSERT HERE: Example preset for new users
example = st.sidebar.selectbox("Try Example Passenger", ["None", "Young Female (1st Class)", "Old Male (3rd Class)"])
if example == "Young Female (1st Class)":
    pclass, sex_input, age, fare = 1, "Female", 22, 100.0
elif example == "Old Male (3rd Class)":
    pclass, sex_input, age, fare = 3, "Male", 65, 10.0

# Convert inputs to numbers
sex = 0 if sex_input == "Male" else 1
embarked = {"C = Cherbourg": 0, "Q = Queenstown": 1, "S = Southampton": 2}[embarked_input]

# Predict button
if st.sidebar.button("Predict Survival"):
    # Convert input to DataFrame (fixes warning)
    input_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=features)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]
    proba = model.predict_proba(input_df)[0]

    # Show survival prediction
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(result)
    st.write(f"Prediction Confidence: {probability*100:.2f}%")

    # 🧩 INSERT HERE: Show both probabilities
    st.write(f"🔵 Probability of Survival: {proba[1]*100:.2f}%")
    st.write(f"⚫ Probability of Not Surviving: {proba[0]*100:.2f}%")

    # ------------------------------
    # Feature contributions (why survived / why not)
    # ------------------------------
    contributions = input_df.iloc[0] * model.coef_[0]
    contrib_df = pd.DataFrame({
        'Feature': features,
        'Value': input_df.iloc[0],
        'Contribution': contributions
    }).sort_values(by='Contribution', key=abs, ascending=False)

################################################################################################
with st.expander("🔍 Explain Prediction", expanded=False):
    st.subheader("🤔 Why did the model predict this?")

    # ------------------------------
    # Find closest real example
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

    # ------------------------------
    # Emoji-based interpretation for each feature
    # ------------------------------
    def feature_effect(feature, value):
        """
        Returns explanation + impact (-1 or +1)
        """
        if feature == "sex":
            if value == "female":
                return "✅ Being **female** greatly increased survival chances.", 1
            else:
                return "❌ Being **male** reduced survival chances.", -1

        elif feature == "pclass":
            if value == 1:
                return "✅ Traveling in **1st class** greatly increased survival chances.", 1
            elif value == 2:
                return "✅ Traveling in **2nd class** moderately increased survival chances.", 1
            else:
                return "❌ **3rd class** reduced survival chances.", -1

        elif feature == "age":
            if value < 18:
                return "✅ Being a **child** increased priority during evacuation.", 1
            else:
                return "❌ Adult or older age reduced survival chances.", -1

        elif feature == "sibsp":
            if value <= 2:
                return "✅ Traveling alone or with few family members helped survival.", 1
            else:
                return "❌ Too many family members reduced evacuation efficiency.", -1

        elif feature == "parch":
            if value <= 2:
                return "✅ Having few or no parents/children was generally fine.", 1
            else:
                return "❌ Many dependents reduced mobility.", -1

        elif feature == "fare":
            if value > 20:
                return "✅ Higher fare often meant better access to lifeboats.", 1
            else:
                return "❌ Low fare linked to poor access to lifeboats.", -1

        elif feature == "embarked":
            if value in ["C", "Q"]:
                return "✅ Cherbourg/Queenstown passengers had slightly higher survival.", 1
            else:
                return "❌ Southampton passengers had lower survival.", -1

    # ------------------------------
    # Evaluate all features
    # ------------------------------
    if isinstance(sex, (int, float)):
        sex = "male" if sex == 0 else "female"

    if isinstance(embarked, (int, float)):
        embarked_map = {0: "C", 1: "Q", 2: "S"}
        embarked = embarked_map.get(embarked, embarked)  # fallback to value if not in map

    feature_inputs = {
        "sex": sex,
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }

    # ------------------------------
    # Feature weights (importance)
    # ------------------------------
    weights = {
        "sex": 3,
        "pclass": 3,
        "age": 2,
        "sibsp": 1,
        "parch": 1,
        "fare": 1,
        "embarked": 1
    }

    # ------------------------------
    # Evaluate features with weights
    # ------------------------------
    score = 0
    for f, v in feature_inputs.items():
        msg, impact = feature_effect(f, v)
        st.write(msg)
        score += impact * weights[f]

    max_score = sum(weights.values())
    survival_prob = (score + max_score) / (2 * max_score) * 100

    # ------------------------------
    # Display survival tendency
    # ------------------------------
    st.write("### 🧭 Overall Survival Tendency:")
    st.progress(survival_prob / 100)

    if survival_prob >= 50:
        st.success(f"🟢 Great chance of survival! ({survival_prob:.1f}%)")
    else:
        st.error(f"🔴 Low chance of survival ({survival_prob:.1f}%)")

    st.caption("💡 Green ✅ means factors that helped survival, red ❌ means they reduced it.")

    # ------------------------------
    # Show closest real passenger
    # ------------------------------
    st.divider()
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

# 🧩 INSERT HERE: Show sample dataset
with st.expander("🧾 View Sample Data"):
    st.dataframe(df.head(10))

    st.markdown("""
    **Feature Descriptions:**
    - **pclass** → Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)  
    - **sex** → Gender (0 = Male, 1 = Female)  
    - **age** → Passenger’s age in years  
    - **sibsp** → Number of siblings or spouses aboard  
    - **parch** → Number of parents or children aboard  
    - **fare** → Ticket price (in British pounds)  
    - **embarked** → Port of embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)  
    - **survived** → Survival status (0 = No, 1 = Yes)
    """)

st.markdown("""
    ---
    ### 🎓 Insights from the Titanic Dataset

    The Titanic followed an **unwritten rule** during evacuation:

    🧍‍♀️ **"Women and children first."**

    Because of this:
    - 👩‍🦰 **Females had a much higher survival rate** than males.  
    - 👶 **Children** were prioritized for lifeboats.  
    - 🎩 **First-class passengers** had better access to lifeboats and deck space.  
    - 💸 **Third-class passengers** (lower decks) faced difficulty reaching lifeboats in time.
    - 👫 **Siblings/Spouses aboard (`sibsp`)** could influence survival — passengers with close family onboard sometimes had **higher chances of being helped**, but very large families could face difficulty evacuating.  
    - 👨‍👩‍👧 **Parents/Children aboard (`parch`)** also mattered — families traveling together could either **increase survival chances** (if helping each other) or **decrease it** (if coordination was difficult in a crisis).  
    - ⚓ **Port of Embarkation (`embarked`)** also played a small role — passengers boarding from **Cherbourg (C)** were often **wealthier**, while those from **Southampton (S)** mostly belonged to **third class**, leading to **different survival rates**.  

    **Patterns observed:**
    - 🟢 Higher survival chances → Female, First Class, Younger age, Cherbourg (C)  
    - 🔴 Lower survival chances → Male, Third Class, Older age, Southampton (S)

    These patterns are clearly visible in the charts below.
    """)

# 🧩 Feature importance visualization
st.subheader("🧠 Feature Importance (Model Coefficients)")

st.markdown("""
---
### 📘 What Does This Mean?

Each feature in the Titanic dataset — like **age**, **sex**, or **pclass** — affects the model's prediction differently.

In this chart:
- 📈 **Positive values** → increase survival chances  
- 📉 **Negative values** → decrease survival chances  

Hover over each bar to see a **plain-language explanation** of its effect.
""")

import altair as alt

# Create DataFrame with feature coefficients
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# Add plain-language interpretation
interpretations = []
for f, c in zip(coef_df['Feature'], coef_df['Coefficient']):
    if f == 'sex':
        interpretations.append("Being female increases survival" if c > 0 else "Being female decreases survival")
    elif f == 'pclass':
        interpretations.append("Higher class increases survival" if c < 0 else "Lower class increases survival")
    elif f == 'age':
        interpretations.append("Older age decreases survival" if c < 0 else "Older age increases survival")
    elif f == 'sibsp':
        interpretations.append("More siblings/spouses decreases survival" if c < 0 else "More siblings/spouses increases survival")
    elif f == 'parch':
        interpretations.append("More parents/children decreases survival" if c < 0 else "More parents/children increases survival")
    elif f == 'fare':
        interpretations.append("Higher fare increases survival" if c > 0 else "Higher fare decreases survival")
    elif f == 'embarked':
        interpretations.append("Boarding port affects survival (Cherbourg/Southampton/Queenstown)")
    else:
        interpretations.append("")
coef_df['Effect'] = interpretations

# Build interactive Altair chart
chart = (
    alt.Chart(coef_df)
    .mark_bar(color='#1f77b4')
    .encode(
        x=alt.X('Feature:N', sort=None, title='Feature'),
        y=alt.Y('Coefficient:Q', title='Coefficient Value'),
        tooltip=[
            alt.Tooltip('Feature:N', title='Feature'),
            alt.Tooltip('Coefficient:Q', format=".4f", title='Effect on Survival'),
            alt.Tooltip('Effect:N', title='Interpretation')
        ]
    )
    .properties(width=600, height=400)
)

st.altair_chart(chart, use_container_width=True)
st.caption("💬 Positive values increase survival chances; negative values decrease them.")

st.markdown("""
---
### 🧠 What is a Confusion Matrix?  

Imagine you are the **captain of a lifeboat assignment team** on the Titanic.  
Your job is to **predict whether each passenger survived or not** based on their details (class, age, gender, family aboard, etc.).  

After the disaster, you compare your **predictions vs reality**:

- Some passengers you correctly predicted would survive ✅  
- Some you correctly predicted would not survive ✅  
- Some you thought would survive but did not ❌  
- Some you thought would not survive but actually survived ❌  

This table of **predictions vs actual outcomes** is exactly what a **Confusion Matrix** shows.

**Analogy:**  
Think of it as a **“scorecard for your predictions”**:

**The 2×2 grid:**

|               | Predicted: Did NOT Survive | Predicted: Survived |
|---------------|---------------------------|-------------------|
| Actual: Did NOT Survive  | ✅ Correctly predicted did NOT survive (True Negative) | ❌ Wrong prediction (False Positive) |
| Actual: Survived | ❌ Wrong prediction (False Negative) | ✅ Correctly predicted survived (True Positive) |

- ✅ Top-left & bottom-right = predictions that were correct  
- ❌ Top-right & bottom-left = predictions that were wrong  

The heatmap diagram you see is a **visual version of this scorecard**:  
- Darker colors = more passengers in that category  
- Helps you **quickly see where the model is accurate and where it makes mistakes**
---
""")

st.subheader("🧠 Model Performance (Interactive Confusion Matrix)")

# Prepare confusion matrix DataFrame
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, columns=['Predicted: Did Not Survive', 'Predicted: Survived'],
                     index=['Actual: Did Not Survive', 'Actual: Survived'])

# Convert to long format for Altair
cm_long = cm_df.reset_index().melt(id_vars='index')
cm_long.columns = ['Actual', 'Predicted', 'Count']

# Add interpretation for hover tooltip
def interpret(row):
    if row['Actual'] == 'Actual: Survived' and row['Predicted'] == 'Predicted: Survived':
        return "✅ True Positive: Correctly predicted survived"
    elif row['Actual'] == 'Actual: Survived' and row['Predicted'] == 'Predicted: Did Not Survive':
        return "❌ False Negative: Predicted did not survive but survived"
    elif row['Actual'] == 'Actual: Did Not Survive' and row['Predicted'] == 'Predicted: Survived':
        return "❌ False Positive: Predicted survived but did not survive"
    else:
        return "✅ True Negative: Correctly predicted did not survive"

cm_long['Interpretation'] = cm_long.apply(interpret, axis=1)

# Build Altair heatmap
heatmap = alt.Chart(cm_long).mark_rect().encode(
    x=alt.X('Predicted:N', title='Predicted'),
    y=alt.Y('Actual:N', title='Actual'),
    color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
    tooltip=['Actual:N', 'Predicted:N', 'Count:Q', 'Interpretation:N']
).properties(width=400, height=300)

st.altair_chart(heatmap, use_container_width=True)

st.markdown("💬 Hover over each cell to see exactly what it represents and how many passengers fall into that category.")

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()  # flatten 2x2 matrix
total = tn + fp + fn + tp
predicted_survived = tp + fp
predicted_not_survived = tn + fn

total_survived = y_test.sum()  # actual survivors
total_not_survived = len(y_test) - total_survived  # actual non-survivors

# Beginner-friendly story summary
st.markdown(f"""
### 📢 Model Prediction Summary

Out of **{total} passengers in the test set**:

- 🟢 **{predicted_survived} passengers were predicted to survive**  
- 🔴 **{predicted_not_survived} passengers were predicted NOT to survive**

Breaking it down:

- 🧍‍♀️ **Total actual survivors:** {total_survived}  
  → ✅ Model correctly predicted **{tp}** of them survived  
  → ❌ Model missed **{fn}** (predicted they didn’t survive)

- ⚰️ **Total actual non-survivors:** {total_not_survived}  
  → ✅ Model correctly predicted **{tn}** did not survive  
  → ❌ Model wrongly predicted **{fp}** of them as survived  

- ✅ **Correctly predicted survived:** {tp}  
- ✅ **Correctly predicted did NOT survive:** {tn}  
- ❌ **Incorrectly predicted survived (but did not):** {fp}  
- ❌ **Incorrectly predicted did NOT survive (but did):** {fn}  

**Overall, the model correctly predicted {(tp+tn)/total*100:.2f}% of passengers.**

💡 This summary helps you see not just numbers, but also the **story of how the model performed** on actual Titanic passengers.
""")

# Survival count by gender (friendly labels)
st.subheader("Survival Count by Gender")

df_gender = df.copy()
df_gender['Survived'] = df_gender['survived'].map({0: 'Did Not Survive', 1: 'Survived'})
df_gender['Sex'] = df_gender['sex'].map({0: 'Male', 1: 'Female'})

chart = alt.Chart(df_gender).mark_bar().encode(
    x='Survived:N',
    y='count()',
    color='Sex:N',
    tooltip=['Survived:N', 'Sex:N', 'count()']
).properties(width=400, height=300)

st.altair_chart(chart, use_container_width=True)
st.caption("💡 Hover over the bars to see exact counts of males and females who survived or did not survive.")

st.caption("""
💡 **What this chart tells us:**

- 🧍‍♀️ **Females had a much higher survival rate** than males.  
  You can see more females in the "Survived" category.  

- 👨 **Males had a lower survival rate**, with most males in the "Did Not Survive" category.  

- This aligns with the **Titanic evacuation rule:** "Women and children first."  
- It also shows how **gender was one of the strongest factors affecting survival**.
""")

# Survival rate by passenger class (friendly labels)
st.subheader("Survival Rate by Passenger Class")

df_class = df.copy()
df_class['Class'] = df_class['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

class_chart = alt.Chart(df_class).mark_bar().encode(
    x='Class:N',
    y='survived:Q',
    tooltip=['Class:N', alt.Tooltip('survived:Q', format=".2f", title='Survival Rate')],
    color='Class:N'
).properties(width=400, height=300)

st.altair_chart(class_chart, use_container_width=True)
st.caption("""
💡 **Insights:**
- 🎩 1st class passengers had the highest survival rate  
- 🏢 2nd class passengers had a moderate survival rate  
- 💸 3rd class passengers had the lowest survival rate  
- Passenger class strongly influenced survival due to **access to lifeboats and deck space**
""")

# ------------------------------
# 3️⃣ Survival rate by embarkation port
# ------------------------------
st.subheader("Survival Rate by Port of Embarkation")

survival_by_port = df.groupby('embarked')['survived'].mean().reset_index()
survival_by_port['Port'] = survival_by_port['embarked'].map({0:'Cherbourg (C)', 1:'Queenstown (Q)', 2:'Southampton (S)'})

embarked_chart = alt.Chart(survival_by_port).mark_bar().encode(
    x='Port:N',
    y=alt.Y('survived:Q', title='Survival Rate'),
    color='Port:N',
    tooltip=['Port:N', alt.Tooltip('survived:Q', format=".2f", title='Survival Rate')]
).properties(width=400, height=300)

st.altair_chart(embarked_chart, use_container_width=True)

st.caption("""
💡 **Insights:**
- ⚓ Passengers boarding from **Cherbourg (C)** had the **highest survival rate**  
- 🚢 Passengers from **Southampton (S)** mostly belonged to **3rd class** → **lower survival rates**  
- 🛳 Passengers from **Queenstown (Q)** had **moderate survival rates**, in between C and S  
- Port of embarkation reflects **socioeconomic differences**, affecting access to lifeboats and survival chances
""")

# Survival rate by age groups
st.subheader("Survival Rate by Age Group")

# Create age bins
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=[f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)])

# Calculate survival rate per age group
age_summary = df.groupby('age_group')['survived'].mean().reset_index()

# Plot
age_chart = alt.Chart(age_summary).mark_bar().encode(
    x='age_group:N',
    y=alt.Y('survived:Q', title='Survival Rate'),
    color='survived:Q',
    tooltip=['age_group:N', alt.Tooltip('survived:Q', format=".2f", title='Survival Rate')]
).properties(width=600, height=300)

st.altair_chart(age_chart, use_container_width=True)

st.caption("""
💡 **Insights:**
- 👶 **Children (0-10)** had the highest survival rates  
- 🧍 Young adults (20-30) survived more often than older adults  
- 👴 **Older passengers (>60)** had the lowest survival rates  
- Age alone isn’t the only factor — **gender, passenger class, and lifeboat access** also strongly influenced survival
""")

# ------------------------------
# 5️⃣ Survival rate by number of siblings/spouses (sibsp)
# ------------------------------
st.subheader("Survival Rate by Number of Siblings/Spouses Aboard")

df_sibsp = df.groupby('sibsp')['survived'].mean().reset_index()

sibsp_chart = alt.Chart(df_sibsp).mark_bar().encode(
    x=alt.X('sibsp:O', title='Number of Siblings/Spouses'),
    y=alt.Y('survived:Q', title='Survival Rate'),
    color=alt.Color('survived:Q', scale=alt.Scale(scheme='blues')),
    tooltip=[alt.Tooltip('sibsp:O', title='Siblings/Spouses'), 
             alt.Tooltip('survived:Q', format=".2f", title='Survival Rate')]
).properties(width=400, height=300)

st.altair_chart(sibsp_chart, use_container_width=True)
st.caption("""
💡 **Insights:**  
- Passengers with **1–2 siblings/spouses** had slightly higher survival chances  
- Traveling alone (0) or in very large groups often decreased survival  
- Family presence sometimes helped passengers get access to lifeboats
""")

# ------------------------------
# 6️⃣ Survival rate by number of parents/children (parch)
# ------------------------------
st.subheader("Survival Rate by Number of Parents/Children Aboard")

df_parch = df.groupby('parch')['survived'].mean().reset_index()

parch_chart = alt.Chart(df_parch).mark_bar().encode(
    x=alt.X('parch:O', title='Number of Parents/Children'),
    y=alt.Y('survived:Q', title='Survival Rate'),
    color=alt.Color('survived:Q', scale=alt.Scale(scheme='greens')),
    tooltip=[alt.Tooltip('parch:O', title='Parents/Children'), 
             alt.Tooltip('survived:Q', format=".2f", title='Survival Rate')]
).properties(width=400, height=300)

st.altair_chart(parch_chart, use_container_width=True)
st.caption("""
💡 **Insights:**  
- Passengers traveling with **1–3 parents/children** generally had higher survival chances  
- Very large groups or traveling alone sometimes decreased survival  
- Shows that family presence influenced survival on the Titanic
""")

# ------------------------------
# 7️⃣ FAQ Section
# ------------------------------
st.header("❓ Frequently Asked Questions (FAQ)")

with st.expander("What if I enter a very young child or a very old passenger?"):
    st.write("""
    - The model uses **age as a numeric input**.  
    - Very young children often have **higher predicted survival** due to historical "women and children first" policy.  
    - Very old passengers may have **lower predicted survival**, but remember, this is a **statistical model**, not a guarantee.
    """)

with st.expander("What if I enter a combination that doesn’t exist in the dataset?"):
    st.write("""
    - The app tries to find the **closest real match**, but sometimes no exact match exists.  
    - The model will still give a prediction, but the **real answer may be unavailable**.
    """)

with st.expander("Why does my prediction sometimes differ from real Titanic history?"):
    st.write("""
    - The model is **trained on historical Titanic data**, but it is a **simplified logistic regression**.  
    - It cannot capture every detail of human behavior or lifeboat access — just general patterns.
    """)

with st.expander("What does the 'prediction confidence' mean?"):
    st.write("""
    - Confidence shows the model's **estimated probability** that a passenger survived or not.  
    - A **high confidence** means the model is more certain about its prediction;  
    - A **low confidence** means the prediction is more uncertain.
    """)

with st.expander("Can this model be used for other ships or disasters?"):
    st.write("""
    - No. This model is **specific to the Titanic dataset**.  
    - Other ships, disasters, or time periods may have very different conditions.
    """)

# ------------------------------
# 8️⃣ Historical / Contextual FAQ Section
# ------------------------------
st.header("📜 Titanic History & Context FAQ")

with st.expander("Why did passengers from Cherbourg (C) have higher survival rates?"):
    st.write("""
    ⚓ Cherbourg passengers were often **wealthier and in 1st class**, giving them **better access to lifeboats**.  
    Port of embarkation reflects **socioeconomic status**, which influenced survival.
    """)

with st.expander("Why did passengers from Southampton (S) have lower survival rates?"):
    st.write("""
    🚢 Most Southampton passengers were **3rd class**, located on **lower decks**, making it harder to reach lifeboats.
    """)

with st.expander("Why do females have higher survival chances than males?"):
    st.write("""
    🧍‍♀️ Historical evacuation followed **“women and children first”**, so females were prioritized for lifeboats.
    """)

with st.expander("Why do children have better survival chances than adults?"):
    st.write("""
    👶 Children were also prioritized during evacuation.  
    Families and crew often **helped children into lifeboats first**.
    """)

with st.expander("Why does traveling alone sometimes decrease survival chances?"):
    st.write("""
    👨‍👩‍👧 Traveling with **1–2 family members** could help passengers get **assistance to lifeboats**.  
    Large groups or complete isolation sometimes led to **difficulty reaching safety**.
    """)

with st.expander("Why do 1st class passengers survive more often than 3rd class?"):
    st.write("""
    🎩 1st class cabins were **closer to lifeboats** and **on upper decks**.  
    3rd class passengers often faced **narrow corridors and delayed evacuation**.
    """)

with st.expander("Why does having too many siblings/spouses or parents/children reduce survival?"):
    st.write("""
    Large groups (e.g., 4+ family members) could be **harder to coordinate**, slowing access to lifeboats.  
    Smaller groups or solo travelers sometimes had **less support**, but **moderate family size helped**.
    """)

with st.expander("Why does age alone not guarantee survival?"):
    st.write("""
    👴 Older passengers often had **lower survival**, but gender, class, and family presence also matter.  
    The model combines all features to make predictions.
    """)

with st.expander("Why is the prediction not always 100% accurate?"):
    st.write("""
    📊 The model is a **simplified logistic regression** trained on historical data.  
    It captures **general trends**, not every individual scenario.
    """)

with st.expander("Why can’t we use this model for other ships or disasters?"):
    st.write("""
    🛳 This model is **specific to the Titanic**.  
    Different ships, crew policies, and lifeboat conditions would produce **very different survival patterns**.
    """)

# ------------------------------
# 9️⃣ Scenario-based & Curiosity-driven FAQ Section
# ------------------------------
st.header("❓ Titanic Survival: Scenario & Curiosity")

# 1️⃣ Scenario-based questions
with st.expander("If a child boy is in 3rd class, will he have a chance of survival?"):
    st.write("""
    👶 Even though children were prioritized, being in **3rd class (lower deck)** made it **harder to reach lifeboats**. 
    Survival is possible but **lower than 1st or 2nd class children**.
    """)

with st.expander("If a female adult is in 1st class, what are her chances?"):
    st.write("""
    🧍‍♀️ Very high — women were prioritized, and **1st class passengers had best lifeboat access**.
    """)

with st.expander("If a male adult is in 1st class, can he survive?"):
    st.write("""
    🎩 Survival is moderate — men were **less prioritized**, but 1st class **made evacuation easier**.
    """)

with st.expander("If a male child is in 1st class, how likely is survival?"):
    st.write("""
    👦 High — children were prioritized, and 1st class location helped **access to lifeboats faster**.
    """)

with st.expander("If someone travels alone in 3rd class, what is their chance?"):
    st.write("""
    ⚰️ Low — being **alone and in 3rd class** reduces help and lifeboat access.
    """)

with st.expander("If a female adult is in 3rd class, can she survive?"):
    st.write("""
    👩 Fair — women had priority, but **3rd class obstacles** could slow evacuation.
    """)

with st.expander("If a child has 2 siblings aboard, does it help?"):
    st.write("""
    🧍‍👧 Slightly — having **family to support** can help reach lifeboats, but **large groups may complicate escape**.
    """)

with st.expander("If an older male is traveling with family in 1st class, can he survive?"):
    st.write("""
    👴 Better than alone — family may help him **get to lifeboats**, but older age slightly reduces survival.
    """)

with st.expander("If a female adult is traveling alone from Southampton, what happens?"):
    st.write("""
    🚢 Moderate — she is female (priority), but **3rd class / lower deck location** can delay evacuation.
    """)

with st.expander("If a child boy in 2nd class boards from Cherbourg, how likely is survival?"):
    st.write("""
    ⚓ Good chance — **middle class + child + Cherbourg port** improves survival chances.
    """)

# 2️⃣ Curiosity-driven questions
with st.expander("Will a 3rd class boy child survive before a 1st class adult male?"):
    st.write("""
    👶 Children were prioritized over adults, so **yes, the child often has higher survival chances**, even in 3rd class, though deck location still matters.
    """)

with st.expander("If a female adult is in 3rd class, can she survive?"):
    st.write("""
    👩 Women had priority, but being in **3rd class (lower decks)** reduces her access to lifeboats, so her survival probability is moderate.
    """)

with st.expander("If a male child is in 1st class, how likely is he to survive?"):
    st.write("""
    👦 Very likely — children were prioritized, and 1st class placement makes lifeboat access faster.
    """)

with st.expander("If a male adult is in 1st class, can he survive?"):
    st.write("""
    🎩 Survival is possible but not guaranteed — men were **less prioritized**, though 1st class gives some advantage.
    """)

with st.expander("If a child has many siblings aboard, does it help or hurt survival?"):
    st.write("""
    🧍‍👧 Small families help — the child has support reaching lifeboats.  
    ❌ Very large families may slow evacuation, reducing chances.
    """)

with st.expander("If a female adult travels alone from Southampton, what happens?"):
    st.write("""
    🚢 She is female (priority), but **3rd class / lower deck location** reduces chances, so survival probability is moderate.
    """)

with st.expander("If an older male is traveling with family in 1st class, can he survive?"):
    st.write("""
    👴 Better than alone — family support helps, and 1st class access is good, but age still slightly reduces chances.
    """)

with st.expander("If a child boy is in 2nd class boarding from Cherbourg, how likely is survival?"):
    st.write("""
    ⚓ Good — 2nd class is better than 3rd, child priority helps, and Cherbourg passengers were usually wealthier.
    """)

with st.expander("If an adult male is traveling with many children in 3rd class, does he survive?"):
    st.write("""
    ❌ Adults are deprioritized, so chances are lower, though helping children may slightly improve his chance.
    """)

with st.expander("Does traveling alone in 3rd class reduce survival?"):
    st.write("""
    ⚰️ Yes — no family to help and **lower deck access** make it harder to reach lifeboats.
    """)

# 🧩 INSERT HERE: Learn more / next steps
with st.expander("🚀 Learn More"):
    st.write("""
    - Try other algorithms like **Decision Tree** or **Random Forest**  
    - Add new features like **family size = sibsp + parch**  
    - Normalize data to improve performance  
    - Deploy your next ML app on **Streamlit Cloud** for free!
    """)
