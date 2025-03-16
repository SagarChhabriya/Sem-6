# Run Command: streamlit run file_name.py

# 1. import library

# import streamlit as st
# import pickle
# import numpy as np

# 2.  Load the trained model

# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)


# 3. Streamlit App UI
# st.title("🎓 CGPA to Salary Package Predictor 💼")


# 4. Input field for CGPA
# cgpa = st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, step=0.1)


# if st.button("Predict Salary Package"):
#     try:
#         # Prepare input for prediction
#         input_data = np.array([[cgpa]])  # Reshape to match training format
        
#         # Make prediction and convert to float
#         # predicted_package = float(model.predict(input_data).flatten()[0])
#         predicted_package = float(model.predict(input_data)[0, 0])

        
#         # Display result
#         st.success(f"📢 Predicted Salary Package: ₹{predicted_package:.2f} LPA")
    
#     except Exception as e:
#         st.error(f"Error: {e}")