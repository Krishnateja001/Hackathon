import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime
import numpy
from diffusers import DiffusionPipeline
from sklearn.preprocessing import LabelEncoder
# 1. Import your model and any necessary dependencies here
if os.path.exists("Testing/commuter_count.joblib"):
    model = joblib.load("Testing/commuter_count.joblib")
if os.path.exists("Testing/ride_price.joblib"):
    model_2 = joblib.load("Testing/ride_price.joblib")   
def yash():
    st.title("Uber Lyft Prices")
    st.write("Enter the source and destination address")
    places = ('Haymarket Square', \
    'Back Bay', 'North End','North Station','Beacon Hill','Boston University', \
    'Fenway','South Station','Theatre District','West End','Financial District','Northeastern University')
    response = {'hour':0,'day':0,'month':0,'source':'','destination':'','cab_type':[]}
    df_response = response
    input_source = st.selectbox('Enter the source?',places )
    input_destination = st.selectbox('Enter the destination?',places)
    response['destination']=input_destination
    response['source']=input_source
    date_1 = st.date_input(label='Please select date')
    time_2 = st.time_input(label='Please select time')
    # response['year'] = date_1.year
    response['month'] = date_1.month
    response['day'] = date_1.day
    response['hour']=time_2.hour
    response['cab_type']=['Lyft','Uber']
    df_response['month'] = date_1.month
    df_response['day'] = date_1.day
    df_response['hour']=time_2.hour
    df_response['cab_type']=['Lyft','Uber']
    df_response['source']=input_source
    df_response['destination']=input_destination
    dataframe_df=pd.DataFrame(df_response)  

    if st.button("Enter"):
        enc1=LabelEncoder()
        enc1.classes_=numpy.load('Testing/class_source.npy',allow_pickle=True)
        enc2=LabelEncoder()
        enc2.classes_=numpy.load('Testing/class_destination.npy',allow_pickle=True)
        enc3=LabelEncoder()
        enc3.classes_=numpy.load('Testing/class_cab.npy',allow_pickle=True)

        dataframe_df['source']=enc1.transform(dataframe_df['source'])
        dataframe_df['destination']=enc2.transform(dataframe_df['destination'])
        dataframe_df['cab_type']=enc3.transform(dataframe_df['cab_type'])
        print(dataframe_df)
        prediction  = model_2.predict(dataframe_df)
        print(prediction)
        # prediction2  = model.predict(var2)
        st.subheader(f"Lyft ,Uber {prediction}")

# 2. Set up your Streamlit app
def main():
    # (Optional) Set page title and favicon.
    st.set_page_config(page_title="Hackathon Model Showcase", page_icon="🧊")

    # (Optional) Set a sidebar for your app.
    with st.sidebar:
        # st.image("IMAGE_PATH")
        st.title("SIDE_BAR_TITLE")
        choice = st.radio(
            "Menu", ["Home", "MBTA","Cab","Diffusion", "Batch Prediction"])
        st.info(
            "PROJECT_DESCRIPTION")
    
    # Now lets add content to each sub-page of your site
    if choice == "Home":
        # Add a title and some text to the app:
        st.title("Daily Commute decision model")
        st.write(
            "Welcome to the Hackathon Model Showcase by Team NP-completing! Enter the necessary input and see smarter commute choices.")
   
    elif choice == "MBTA":
        st.title("MBTA line")
        st.write("choose your T line")
        purplelineroutes={'year':0,'month':0,'day':0,'hour':0,'Middleborough/Lakeville':0,'Lowell':0,'Haverhill':0,'Kingston':0,'Needham':0,'Fitchburg':0,
                          'Greenbush':0,'Fairmount':0,'Providence/Stoughton':0,'Newburyport/Rockport':0,'Framingham/Worcester':0,
                          'Franklin/Foxboro':0}
        lables = purplelineroutes.keys() - ['year','month','day','hour']
        selectedoption = st.selectbox(label='LINE',options=lables)
        purplelineroutes[selectedoption] = 1
        date_selected = st.date_input(label='Please select date')
        time_selected = st.time_input(label='Please select time')
        
        purplelineroutes['year'] = date_selected.year
        purplelineroutes['month'] = date_selected.month
        purplelineroutes['day'] = date_selected.day
        purplelineroutes['hour']=time_selected.hour
        print(purplelineroutes)
        df = pd.DataFrame(purplelineroutes,index=[0])
        # print(df)
        pop_index = ''
        if st.button("Predict"):
            print(purplelineroutes)
            prediction  = model.predict(df)
            print('I AM HERE')
            print(prediction)
            st.subheader(f"Number of people that will be approximately boarding on the purple line {selectedoption} are {prediction} on {date_selected}")
            prediction = list(prediction)
            prediction = float(prediction[0])
            print('HERE',prediction)
            if prediction >= 4000.0:
                pop_index ="high"
            else:
                pop_index="low"
            print(pop_index)
            st.subheader(f"Crowd index: {pop_index}")
        #d = datetime.strptime(date_selected+' '+time_selected,'%Y-%m-%d %H:%m:s')
       # print(d)
    elif choice == "Cab":
        yash()

    elif choice == "Diffusion":
        # Add a title and some text to the app:
        st.title("Noise Diffusion")
        st.write("Enter the text:")

        # Add your input fields here
        # For example:
        input_text = st.text_input("Enter text for image")

        # Add a button to trigger the prediction
        if st.button("Predict"):
            # Call your model and perform the prediction here
            # For example:
            # prediction = _singlePredict(input_text)
            ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
            prompt = input_text
            images = ldm([prompt], num_inference_steps=10, eta=0.3, guidance_scale=6).images
            st.image(images[0])
            

            # Display the prediction result
            # st.subheader(f"Prediction: {prediction}")

    elif choice == "Batch Prediction":
        # Add a title and some text to the app:
        st.title("Batch Prediction")
        st.write("Upload a CSV file and see smarter commute choices.")

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
