### New App For Work Wind Turbine Classify ###
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

#### Page Config ###
st.set_page_config(
    page_title="Wind Whisper",
    page_icon="https://static.vecteezy.com/system/resources/previews/005/562/131/non_2x/wind-turbines-icon-wind-power-plant-alternative-energy-industry-renewable-and-clean-energy-electricity-generation-vector.jpg",
    menu_items={
        "Get help": "mailto:hikmetemreguler@gmail.com",
        "About": "For More Information\n" + "https://github.com/HikmetEmre/Wind_Turbine_Checker"
    }
)

### Title of Project ###
st.title("**:red[WindSense: Turbine Health Prediction]**")

### Markdown ###
st.markdown("**Introducing :red[Wind Whisper] Empowering Turbine Health Classification using Sensory Data and Logistic Regression **.")

### Adding Image ###
st.image("https://raw.githubusercontent.com/HikmetEmre/Wind_Turbine_Checker/main/f_turbine.jpg")

st.markdown("**The :red[Wind Whisper] dataset, made possible by the :red[US Department of Energy] (DOE) in collaboration with the :red[National Renewable Energy Laboratory] (NREL), comprises sensor data from wind turbines. This dataset has been carefully collected and curated to enable the development of a powerful classification model for predicting the health condition of wind turbines. The dataset's attributes represent various sensory readings from wind turbines, while the target variable indicates the turbines' condition as either healthy or in need of maintenance. By acknowledging the DOE/NREL, we express our gratitude for their contribution to the advancement of renewable energy research and safety measures in the wind energy sector.**")

st.markdown("**The :red[Wind Whisper] dataset comprises a total of 9.6 million sensor data samples collected from wind turbines. Each data sample includes various readings from sensors installed on wind turbines, capturing important metrics related to their health and performance.**")

st.markdown("**The dataset has been carefully labeled, with approximately 50% of the samples corresponding to wind turbines that are considered damaged or in need of maintenance, and the other 50% representing healthy wind turbines.**")

st.markdown("**The :red[Wind Whisper] dataset and the classification model derived from it play a crucial role in advancing renewable energy research and promoting the reliable and sustainable operation of wind turbines in a rapidly evolving energy landscape**")

st.markdown("*:red[Hay lay up and feel the Wind Whisper's embrace, as it raises the winds of innovation, propelling us forward in our journey to a greener horizon.]*")

st.image("https://raw.githubusercontent.com/HikmetEmre/Wind_Turbine_Checker/main/s_turbine.jpg")

#### Header and definition of columns ###
st.header("**META DATA**")

st.image("https://raw.githubusercontent.com/HikmetEmre/Wind_Turbine_Checker/main/meta1.png")


st.image("https://raw.githubusercontent.com/HikmetEmre/Wind_Turbine_Checker/main/meta2.png")



### Example DF ON STREAMLIT PAGE ###
df=pd.read_csv('for_meta.csv')


### Example TABLE ###
st.table(df.sample(5, random_state=17))



#---------------------------------------------------------------------------------------------------------------------

### Sidebar Markdown ###
st.sidebar.markdown("**INPUT** , **:red[Sensation Data] Below & See The Condition of Turbine!")

### Define Sidebar Input's ###
AN3 = st.sidebar.number_input("**:red[AN3 Value]**")
AN4 = st.sidebar.number_input("**:red[AN4 Value]**")
AN5 = st.sidebar.number_input("**:red[AN5 Value]**")
AN6 = st.sidebar.number_input("**:red[AN6 Value]**")
AN7 = st.sidebar.number_input("**:red[AN7 Value]**")
AN8 = st.sidebar.number_input("**:red[AN8 Value]**")
AN9 = st.sidebar.number_input("**:red[AN9 Value]**")
AN10 = st.sidebar.number_input("**:red[AN10 Value]**")
Speed = st.sidebar.number_input("**:blue[Speed Value]**")



#---------------------------------------------------------------------------------------------------------------------

### Recall Model ###
from joblib import load

log_model = load('logistic_regression_model.pkl')

scaler_model = load('data_scaler.pkl')

input_df = [[AN3,AN4,AN5,AN6,AN7,AN8,AN9,AN10,Speed]]
    






input_df_scaled = scaler_model.transform(input_df)

pred = log_model.predict(input_df_scaled)






#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

### Result Screen ###
if st.sidebar.button("Submit"):

    ### Info message ###
    st.info("You can find the result below.")

    ### Inquiry Time Info ###
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    ### For showing results create a df ###
    results_df = pd.DataFrame({
    'Date': [today],
    'Time': [time],
    'AN3': [AN3],
    'AN4': [AN4],
    'AN5' :[AN5],
    'AN6' :[AN6],
    'AN7': [AN7],
    'AN8': [AN8],
    'AN9': [AN9],
    'AN10': [AN10],
    'Speed': [Speed],
    'Turbine Condition': [pred]
    })

   


    st.table(results_df)

    if pred == 1:
        st.markdown('**:blue[Healthy]**')
        st.image("https://raw.githubusercontent.com/HikmetEmre/Wind_Turbine_Checker/main/healthy.jpg")

    elif pred == 0:
        st.markdown("**:red[Damaged]**")
        st.image("https://raw.githubusercontent.com/HikmetEmre/Wind_Turbine_Checker/main/dmgd.jpg")
      
else:
    st.markdown("Please click the *Submit Button*!")
