import streamlit as st
import joblib
import pandas as pd
import os

# 1. Import your model and any necessary dependencies here
if os.path.exists("Testing/model.joblib"):
    model = joblib.load("Testing/model.joblib")


# 2. Set up your Streamlit app
def main():
    # (Optional) Set page title and favicon.
    st.set_page_config(page_title="Hackathon Model Showcase", page_icon="🧊")

    # (Optional) Set a sidebar for your app.
    with st.sidebar:
        # st.image("IMAGE_PATH")
        st.title("SIDE_BAR_TITLE")
        choice = st.radio(
            "Menu", ["Home", "Single Prediction", "Batch Prediction"])
        st.info(
            "PROJECT_DESCRIPTION")
    
    # Now lets add content to each sub-page of your site
    if choice == "Home":
        # Add a title and some text to the app:
        st.title("Hackathon Model Showcase")
        st.write(
            "Welcome to the Hackathon Model Showcase! Enter the necessary input and see live predictions.")

    elif choice == "Single Prediction":
        # Add a title and some text to the app:
        st.title("Single Prediction")
        st.write("Enter the necessary input and see live predictions.")

        # Add your input fields here
        # For example:
        input_text = st.text_input("Enter text for prediction")

        # Add a button to trigger the prediction
        if st.button("Predict"):
            # Call your model and perform the prediction here
            # For example:
            prediction = _singlePredict(input_text)

            # Display the prediction result
            st.subheader(f"Prediction: {prediction}")

    elif choice == "Batch Prediction":
        # Add a title and some text to the app:
        st.title("Batch Prediction")
        st.write("Upload a CSV file and see live predictions.")

        # Add a file uploader to upload a CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        # If a file is uploaded, process and display predictions
        if uploaded_file is not None:
            try:
              df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error("Error: Invalid CSV file. Please upload a valid CSV file.")
            # Display the uploaded data
            st.subheader("Input Data")
            st.dataframe(df, use_container_width=True)

            # Perform predictions on the uploaded data
            predictions = _batchPredict(df)

            # Display the prediction results
            st.subheader("Prediction Results")
            st.dataframe(predictions, use_container_width=True)

# Define your model prediction function here
# For example:

# We are going to use st.cache to improve performance for predictions.
@st.cache_data
def _singlePredict(input_text):
    # Format the input_text so that you can pass it to the model
    # For example:
    # Call your model to make predictions on the input_text
    # For example:
    prediction = model.predict([[float(input_text)]])

    # Make sure to return the prediction result
    return prediction[0][0]

@st.cache_data
def _batchPredict(df):
    # Format the dataframe so that you can pass it to the model
    # For example:
    df = df[["Temperature"]]

    # Call your model to make predictions on the dataframe
    # For example:
    predictions = model.predict(df)

    # Predictions DF
    dfPredictions = pd.DataFrame(predictions, columns=(["Humidity 💦"]))

    # Make sure to return the prediction results
    return dfPredictions


# Run the app
if __name__ == "__main__":
    main()
