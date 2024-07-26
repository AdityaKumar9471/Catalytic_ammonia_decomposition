import pandas as pd
df= pd.read_excel("catalysis sheet.xlsx")
df.drop(columns="DOI",inplace=True)

from sklearn.preprocessing import LabelEncoder
print(df["Supported by"])
le = LabelEncoder()

df['Catalyst'] = le.fit_transform(df['Catalyst'])
le2=LabelEncoder()
df["Supported by"] = le2.fit_transform(df["Supported by"])

X=df.drop(columns="Ammonia conversion (%)")
y=df["Ammonia conversion (%)"]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.001)
from xgboost import XGBRegressor

model=XGBRegressor()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,y_pred,squared=False))

import streamlit as st
st.set_page_config(page_title="Ammonia Conversion Prediction", layout="centered")
st.title("Ammonia Conversion Prediction")

#st.image("/Users/adityakumar/PycharmProjects/pythonProject1/laboratory-3d-glassware.jpg", use_column_width=True)
st.markdown("""
This application predicts the catalytic decomposition of ammonia based on input reaction conditions and catalytic features.
Utilizing an XGBoost regression model, it applies advanced machine learning techniques to provide accurate predictions of ammonia conversion rates.
""")
# Add some beautifications
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1f4e78;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        font-size: 500px;
        margin: 4px 2px;
        border-radius: 12px;
    }
    .stTextInput label, .stNumberInput label {
        font-size: 500px;
        color: white;
    }
    .stTextInput input, .stNumberInput input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        margin-bottom: 10px;
        background-color: #f5f5f5;
        color: black;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

catalyst_names = [
    "Ni-Co-La",
    "Platinum",
    "Ni2Mo3N",
    "Ni-Ce",
    "Co-Sm-O",
    "CoMONx",
    "Ni-Ce-Al-O",
    "Ru-K",
    "Ruthenium",
    "MoO3",
    "Ni-Co",
    "Co-Mo-Fe-Ni-Cu",
    "Co-Mo",
    "Mo2N",
    "Fe2-N",
    "Fe",
    "Ru-Fe",
    "Nickel",
    "Co3O4",
    "Copper-Zinc",
    "Cu-Zn",
    "Fe-Mo",
    "Cobalt",
    "Nix-Co10-x",
    "MoS2",
    "Co3Mo3N"
]

supporter_names=['Activated clay',
 'Al2O3',
 'Alkali-silicates',
 'Alkaline earth metal aluminate',
 'Alumina',
 'BHA',
 'CNT',
 'Ca-Al LDH',
 'CaO',
 'Carbon',
 'Ce-Mg doped',
 'Ce0.6Zr0.3Y0.1O2',
 'CeO2',
 'CeO2-TiO2',
 'Citric acid',
 'Co-Mo-Fe-Ni-Cu',
 'Co3Mo3N',
 'Cr2O3',
 'Fe2-N',
 'Graphene-aerogel',
 'LHA',
 'La2O2CO3- Al2O3',
 'La2O3',
 'Lanthania-Seria',
 'MCF 17.00',
 'MCM-41',
 'Mg-Al',
 'Mg-La',
 'MgAl2O4',
 'MgO',
 'Mica',
 'MoO3',
 'NC',
 'Na-ZSM-5 Zeolite',
 'Ni-Ce-Al-O',
 'Ni2Mo3N',
 'SBA-15',
 'Self',
 'SiO2',
 'Silica',
 'Siliceous',
 'Sm2O3',
 'Unknown',
 'Y2O3',
 'ZSM-5',
 'Zr doping',
 'ZrO2']
catalyst = st.selectbox("Select Catalyst",options=catalyst_names)
temperature = st.number_input("Reaction Temperature")
Loading = st.number_input("Catalyst total metal loading(%)")
crystal_size = st.number_input("Catalyst average crystallite size (nm)")
crystal_index = st.number_input("Catalyst crystallinity index (-)")
surface_area = st.number_input("Catalyst specific surface area (m².g⁻¹)")
pore_volume = st.number_input("Catalyst pore volume (cm³.g⁻¹)")
pore_dis = st.number_input("Catalyst average pore diameter (nm)")
vel = st.number_input("Gas hourly space velocity (mL.h⁻¹.gcat⁻¹)")
supported_by = st.selectbox("Supporting material of catalyst",options=supporter_names)

# Encode the categorical inputs
if supported_by:
    catalyst_encoded = le.transform([catalyst])[0]
    supported_by_encoded = le2.transform([supported_by])[0]

# Prepare the input for prediction
if supported_by:
    input_data = [[Loading,crystal_size,crystal_index,surface_area,pore_volume,pore_dis,temperature,vel
               ,catalyst_encoded,supported_by_encoded]]
    input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Ammonia Conversion (%): {prediction[0]}")

# Add an image at the bottom
#st.image("/Users/adityakumar/PycharmProjects/pythonProject1/laboratory-glassware-with-colorful-liquid-dark-background-3d-illustration.jpg", use_column_width=True)


