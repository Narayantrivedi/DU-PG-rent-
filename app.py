import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# 1. Create a tiny mock dataset of DU PGs
data = {
    'Distance_Km': [0.5, 1.5, 0.2, 2.0, 1.0, 0.3],
    'Rooms': [1, 2, 1, 3, 2, 1],
    'AC_Included': [1, 0, 1, 0, 1, 0],
    'Rent': [12000, 8500, 14000, 7000, 10000, 9000],
}
df = pd.DataFrame(data)

# 2. Train a quick ML Model (Linear Regression for Rent Prediction)
X = df[['Distance_Km', 'Rooms', 'AC_Included']]
y = df['Rent']
model = LinearRegression()
model.fit(X, y)

# 3. Streamlit User Interface
st.title('🏠 DU PG Smart Finder & Rent Predictor')
st.write(
    'A quick tool for Delhi University students to estimate fair PG rent and find options.'
)

# Sidebar for Navigation
option = st.sidebar.selectbox(
    'Choose Feature', ['Predict PG Rent', 'Browse Mock PGs']
)

if option == 'Predict PG Rent':
  st.subheader('Machine Learning Rent Estimator')
  st.write(
      'Enter the specifications of a PG to predict its fair monthly price.'
  )

  dist = st.slider('Distance from College (km)', 0.1, 3.0, 1.0)
  rooms = st.selectbox(
      'Sharing Type (1 = Single, 2 = Double, 3 = Triple)', [1, 2, 3]
  )
  ac = st.radio('AC Included?', ['Yes', 'No'])
  ac_val = 1 if ac == 'Yes' else 0

  if st.button('Predict Rent'):
    prediction = model.predict([[dist, rooms, ac_val]])[0]
    st.success(
        f'Estimated Fair Rent for this PG: ₹ {int(prediction)} per month'
    )

elif option == 'Browse Mock PGs':
  st.subheader('Available PGs Near Campus')
  budget = st.slider('Max Budget (₹)', 5000, 15000, 10000)

  filtered_df = df[df['Rent'] <= budget]
  st.write(f'Showing PGs under ₹{budget}:')
  st.dataframe(filtered_df)