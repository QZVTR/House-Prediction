import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Cali House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction")
st.write("This app predicts the **median house value** based on various features using xgboost.")

# Form validation function
def formValidation(input_data, inland, near_bay, near_ocean):
    if input_data['total_rooms'] <= 0 or input_data['population'] <= 0 or input_data['households'] <= 0:
        st.error("Total rooms, population, and households must be greater than zero.")
        return False, None

    if not (inland or near_bay or near_ocean):
        st.error("At least one of 'Inland', 'Near Bay', or 'Near Ocean' must be selected.")
        return False, None

    if inland and (near_bay or near_ocean):
        st.error("If 'Inland' is selected, 'Near Bay' and 'Near Ocean' must not be selected.")
        return False, None

    return True, input_data

st.subheader("Input Features")
# Streamlit form
with st.form(key='input_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Location**")
        longitude = st.number_input("Longitude", value=-122.23, step=0.01)
        latitude = st.number_input("Latitude", value=37.88, step=0.01)
    
    with c2:
        st.write("**House Features**")
        housing_median_age = st.number_input("Housing Median Age", value=41.0, step=0.5)
        total_rooms = st.number_input("Total Rooms", value=880, step=10)
        total_bedrooms = st.number_input("Total Bedrooms", value=130, step=10)

    with c3:
        st.write("**Area Features**")
        population = st.number_input("Population", value=322, step=10)
        households = st.number_input("Households", value=126, step=1)
        median_income = st.number_input("Median Income (1 = $1,000)", value=8.3252, step=0.1)
    
    st.write("**Ocean Proximity**")
    inland = st.checkbox("Inland", value=False)
    near_bay = st.checkbox("Near Bay", value=True)
    near_ocean = st.checkbox("Near Ocean (<1H)", value=False)

    location_df = pd.DataFrame({
        'lat': [latitude],
        'lon': [longitude]
    })

    st.write("**Location on Map**")
    # Display map with marker
    st.map(location_df, zoom=5)

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        input_data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": 0 if inland else (1 if near_bay else (2 if near_ocean else 3))
            #'bedrooms_per_room': total_rooms / households if households > 0 else 0,
            #'ocean_proximity_1H_OCEAN': near_ocean,
            #'ocean_proximity_INLAND': inland,
            #'ocean_proximity_ISLAND': False,
            #'ocean_proximity_NEAR_BAY': near_bay,
            #'ocean_proximity_NEAR_OCEAN': near_ocean,
            # For validation purposes:
        }

        valid, data = formValidation(input_data, inland, near_bay, near_ocean)
        #st.info(data)
        if valid:
            # Remove extra validation fields
            #for key in ['total_rooms', 'population', 'households']:
            #    del data[key]

            st.info("Sending data to the prediction API...")
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=data)
                response.raise_for_status()
                prediction = response.json().get("predicted_value")
                st.success(f"🏡 Predicted Median House Value: **${prediction:,.2f}**")
            except requests.RequestException as e:
                st.error(f"❌ API request failed: {e}")


st.markdown("---")
st.caption("Developed by Edward Jermyn")