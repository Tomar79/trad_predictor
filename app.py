import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the saved model
model = load_model("Final_model54.h5")

# Load the dataset
df = pd.read_csv("Boom.csv", delimiter='\t')

# Convert date and time columns to datetime
df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])

# Sort the data by datetime
df.sort_values('datetime', inplace=True)

# Set datetime as the index
df.set_index('datetime', inplace=True)

# Select the relevant columns for modeling
data = df[['<OPEN>', '<HIGH>', '<CLOSE>']]

# Perform Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Update the dataset with three new values
new_open_value= st.number_input(label='Open',step=1.,format="%.4f")
new_high_value= st.number_input(label='High',step=1.,format="%.4f")
new_close_value= st.number_input(label='Close',step=1.,format="%.4f")
Predict=st.button("Predict Next values")
if Predict:
# new_open_value=9892.7190
# new_high_value=9925.7490	
# new_close_value=9888.5020
        new_values = np.array([[new_open_value, new_high_value, new_close_value]])
        new_values = scaler.transform(new_values)
        scaled_data = np.concatenate((scaled_data, new_values))

        # Generate the input sequence for prediction
        time_step = 54
        last_sequence = scaled_data[-time_step:]
        input_sequence = np.array([last_sequence])

        # Make a prediction for the next value
        next_sequence = model.predict(input_sequence)
        next_value = scaler.inverse_transform(next_sequence)[0]
        st.markdown(f"Predicted Open Price:{ next_value[0]}")
        st.markdown(f"Predicted High Price:{ next_value[1]}")        
        st.markdown(f"Predicted Close Price:{ next_value[2]}")


        # # Print the predicted values
        # print("Predicted Open Price:", next_value[0])
        # print("Predicted High Price:", next_value[1])
        # print("Predicted Close Price:", next_value[2])