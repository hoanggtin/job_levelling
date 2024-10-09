import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
import os

# Load pre-trained model function
def load_model():
    with open('jobleveling.pkl', 'rb') as model_file:
        return pickle.load(model_file)

# Function to save prediction history
# Function to save prediction history
def save_history(input_data, prediction):
    # Create the history.csv file if it doesn't exist
    if not os.path.exists('history.csv'):
        history_df = pd.DataFrame(columns=['datetime', 'title', 'location', 'description', 'function', 'industry', 'predicted_career_level'])
        history_df.to_csv('history.csv', index=False)

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a new entry as a DataFrame
    new_entry = pd.DataFrame({
        'datetime': [current_time],
        'title': [input_data['title'][0]],
        'location': [input_data['location'][0]],
        'description': [input_data['description'][0]],
        'function': [input_data['function'][0]],
        'industry': [input_data['industry'][0]],
        'predicted_career_level': [prediction]
    })

    # Read the existing history.csv file
    history_df = pd.read_csv('history.csv')

    # Concatenate the new entry with the existing DataFrame
    history_df = pd.concat([history_df, new_entry], ignore_index=True)

    # Save the updated DataFrame back to history.csv
    history_df.to_csv('history.csv', index=False)


# Function to display prediction history
def show_history():
    if os.path.exists('history.csv'):
        history_df = pd.read_csv('history.csv')
        st.write(history_df)
    else:
        st.write("No history available.")

# Title of the app
st.title('Job Leveling Prediction App')

# Input fields
title = st.text_input('Enter Job Title:')
location = st.text_input('Enter Location:')
description = st.text_area('Enter Job Description:')
function = st.text_input('Enter Job Function:')
industry = st.text_input('Enter Industry:')

# Button to trigger prediction
if st.button('Predict Career Level'):
    # Load the model
    model = load_model()
    
    # Create a dataframe from user input
    input_data = pd.DataFrame({
        'title': [title],
        'location': [location],
        'description': [description],
        'function': [function],
        'industry': [industry]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display the prediction result
    st.success(f'Predicted Career Level: {prediction}')
    
    # Save the prediction to history
    save_history(input_data, prediction)

# Button to show prediction history
if st.button('Show History'):
    show_history()
