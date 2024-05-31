import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pydeck as pdk
from geopy.geocoders import Nominatim
import os

# Define the latitude and longitude ranges for India
india_lat_range = [8, 37]
india_lon_range = [68, 98]

regions = {
    'North': [30, 90, 10, 40],
    'South': [8, 77, 8, 35],
    'East': [21, 90, 8, 24],
    'West': [20, 75, 22, 68]
}

# Function to determine the region based on latitude and longitude
def get_region(latitude, longitude):
    for region, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return region
    return 'Unknown'

# Function to preprocess the data
def preprocess_data(data):
    data['Year'] = pd.to_datetime(data['Origin Time']).dt.year
    data.drop(columns=['Origin Time'], inplace=True)
    data.dropna(inplace=True)
    return data

# Function to calculate probabilities based on occurrences of nearly similar longitude and latitude
def calculate_probabilities(data):
    threshold = 0.1  
    data['Rounded_Longitude'] = data['Longitude'].round(decimals=1)
    data['Rounded_Latitude'] = data['Latitude'].round(decimals=1)
    
    # Count occurrences of nearly similar longitude and latitude combinations
    location_counts = data.groupby(['Rounded_Longitude', 'Rounded_Latitude']).size().reset_index(name='Occurrence_Count')

    location_counts['Probability'] = location_counts['Occurrence_Count'] / location_counts['Occurrence_Count'].sum()

    data = pd.merge(data, location_counts[['Rounded_Longitude', 'Rounded_Latitude', 'Probability']], 
                    on=['Rounded_Longitude', 'Rounded_Latitude'], how='left')
    
    return data

def train_model(data):
    X = data[['Longitude', 'Latitude', 'Magnitude', 'Year', 'Probability']]
    y = data['Probability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Function to plot earthquakes on map
def plot_earthquakes_map(data, selected_places):
    st.pydeck_chart(
        pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=20.5937,
                longitude=78.9629,
                zoom=5,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,
                    get_position='[Longitude, Latitude]',
                    get_radius=10000,
                    get_fill_color=[255, 0, 0],
                    pickable=True,
                    auto_highlight=True,
                ),
            ],
        )
    )

# Function to get place name based on latitude and longitude
def get_place_name(latitude, longitude):
    geolocator = Nominatim(user_agent="earthquake_prediction_app")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    return location.address if location else "Unknown"

# Function to display the Streamlit app
def main_page():
    st.title("Earthquake Prediction in India")
    st.markdown("""
    This website aims to provide insights into earthquake prediction in India.
    """)
    st.markdown("---")
    st.header("Understanding Earthquakes")
    st.markdown("""
    Earthquakes are natural phenomena that occur due to the sudden release of energy in the Earth's crust, causing seismic waves. 
    They can result in widespread devastation and loss of life. Predicting earthquakes can help mitigate their impact and save lives.
    """)
    st.header("Predictions Based on Past Data")
    st.markdown("""
    Our predictions are based on historical earthquake data, including factors such as location, magnitude, and time. 
    By analyzing this data, we aim to identify patterns and trends that can help forecast future earthquake occurrences.
    """)
    
    # Load data
    data = pd.read_csv('Indian_earthquake_data.csv')

    # Filter data for India only
    data = data[(data['Latitude'] >= india_lat_range[0]) & (data['Latitude'] <= india_lat_range[1]) &
                (data['Longitude'] >= india_lon_range[0]) & (data['Longitude'] <= india_lon_range[1])]
    
    # Preprocess the data
    data['Region'] = data.apply(lambda row: get_region(row['Latitude'], row['Longitude']), axis=1)

    # Show the distribution of earthquakes among regions in India
    st.subheader('Distribution of Earthquakes Among Regions in India')
    indian_data = data[data['Location'].str.contains('India', case=False, na=False)]
    region_counts = indian_data['Region'].value_counts()
   

    # Display top 20 earthquake-prone places in India
 
    top_places = indian_data.groupby(['Location']).size().nlargest(20)
    st.bar_chart(top_places.sort_values(ascending=True))

    # Add Explore button
    explore_button = st.button("Explore")
    if explore_button:
        st.empty()  # Clear the content above the button
        run_new_model()

# Function to run the New Model content
def run_new_model():
    st.title('New Model for Earthquake Prediction')

    # Load data
    data = pd.read_csv('Indian_earthquake_data.csv')
    
    # Filter data for India only
    data = data[(data['Latitude'] >= india_lat_range[0]) & (data['Latitude'] <= india_lat_range[1]) &
                (data['Longitude'] >= india_lon_range[0]) & (data['Longitude'] <= india_lon_range[1])]
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Calculate probabilities based on occurrences of nearly similar longitude and latitude
    data = calculate_probabilities(data)
    
    # Train the machine learning model
    model = train_model(data)
    
    # Predict earthquake probabilities for future years (2024 onwards)
    future_years = range(2024, 2030)  # Adjust as needed
    future_data = pd.DataFrame(columns=['Longitude', 'Latitude', 'Magnitude', 'Year']) # Define columns
    for year in future_years:
        future_data.loc[len(future_data)] = [78.9629, 20.5937, 6.0, year]  # Average values
    future_data = calculate_probabilities(future_data)
    future_data.fillna(0, inplace=True)  # Fill missing probabilities with 0
    
    # Predict probabilities for future data
    future_data['Probability'] = model.predict(future_data[['Longitude', 'Latitude', 'Magnitude', 'Year', 'Probability']])
    
    # Display top 20 earthquake-prone places in India
   

    # Plot earthquakes on map
    st.subheader('Earthquake-Prone Places Marked')
    top_places = data.groupby(['Longitude', 'Latitude']).size().nlargest(20).index.tolist()
    plot_earthquakes_map(data, top_places)

    # Display top 20 most earthquake-prone places in the sidebar
    st.sidebar.subheader('Top 20 Most Earthquake-Prone Places')
    for lon, lat in top_places:
        place_name = get_place_name(lat, lon)
        st.sidebar.write(f"Location: {place_name}, Latitude: {lat}, Longitude: {lon}")

if __name__ == "__main__":
    main_page()
