import streamlit as st
import pandas as pd
import joblib
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
import requests

# Load pre-trained models
cost_model = joblib.load("cost_model.pkl")
resource_model = joblib.load("resource_model.pkl")


def get_coordinates_from_postcode(postcode):
    """
    Converts a postcode to latitude and longitude.

    Parameters:
    - postcode (str): The postcode to convert.

    Returns:
    - (float, float): Latitude and longitude if found, otherwise (None, None).
    """
    geolocator = Nominatim(user_agent="store_locator")
    location = geolocator.geocode(postcode)
    if location:
        return location.latitude, location.longitude
    else:
        st.error("Unable to find coordinates for the given postcode.")
        return None, None


def get_nearby_stores(current_lat, current_lon, stores_df, distance_threshold, max_stores):
    """
    Recommend a list of nearby stores within a given distance threshold.
    Excludes stores that are exactly 0 miles from the current location.
    """
    distances = []
    for _, row in stores_df.iterrows():
        store_lat = row['latitude']
        store_lon = row['longitude']
        distance = geodesic((current_lat, current_lon), (store_lat, store_lon)).miles
        distances.append(distance)

    stores_df['Distance'] = distances
    # Filter stores greater than 0 miles away and within the distance threshold
    nearby_stores = stores_df[(stores_df['Distance'] > 0) & (stores_df['Distance'] <= distance_threshold)]
    nearby_stores = nearby_stores.sort_values(by='Distance').head(max_stores)

    return nearby_stores[['Karcher reference', 'latitude', 'longitude', 'Distance']]

import requests

def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    """
    Get road route between two points using OSRM.

    Parameters:
    - start_lat (float): Latitude of the start location.
    - start_lon (float): Longitude of the start location.
    - end_lat (float): Latitude of the end location.
    - end_lon (float): Longitude of the end location.

    Returns:
    - list of (lat, lon) tuples representing the route.
    """
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)

    if response.status_code == 200:
        data = response.json()
        route = data['routes'][0]['geometry']['coordinates']
        # OSRM returns coordinates as (lon, lat), so we need to reverse them
        route = [(lat, lon) for lon, lat in route]
        return route
    else:
        st.error("Error fetching route from OSRM.")
        return []


import folium


def plot_map_with_routes(current_lat, current_lon, nearby_stores):
    """
    Plots a map with road routes from the current location to each nearby store.

    Parameters:
    - current_lat (float): Latitude of the current location.
    - current_lon (float): Longitude of the current location.
    - nearby_stores (DataFrame): DataFrame of nearby stores with 'Latitude', 'Longitude', and 'Store Id'.

    Returns:
    - folium.Map: Folium map object with markers and routes.
    """
    m = folium.Map(location=[current_lat, current_lon], zoom_start=12)
    folium.Marker(
        [current_lat, current_lon],
        popup="Current Location",
        icon=folium.Icon(color='blue', icon='shopping-basket', prefix='fa')
    ).add_to(m)

    for _, row in nearby_stores.iterrows():
        store_lat = row['latitude']
        store_lon = row['longitude']
        distance = row['Distance']

        # Fetch the actual road route from OSRM
        route = get_osrm_route(current_lat, current_lon, store_lat, store_lon)

        # Plot the route on the map if found
        if route:
            folium.PolyLine(
                locations=route,
                color="blue",
                weight=2.5,
                opacity=0.8
            ).add_to(m)

        # Add a marker for each nearby store
        folium.Marker(
            [store_lat, store_lon],
            popup=f"Store ID: {row['Karcher reference']}, Distance: {distance:.2f} miles",
            icon=folium.Icon(color='green', icon='fa-shopping-basket', prefix='fa')
        ).add_to(m)

    return m


def predict1():
    st.title("Coop Store Service Prediction and Nearby Store Recommendation")

    store_size = st.number_input("Store Size (SQ/F)", min_value=500, max_value=5000, value=3000, step=100)
    productivity = st.number_input("Productivity (SQ/F per hour)", min_value=50, max_value=500, value=200, step=10)
    demand_score = st.slider("Demand Score", 1, 10, 5)
    priority_level = st.selectbox("Priority Level", [1, 2, 3],
                                  format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High'}[x])

    # Get postcode from user
    postcode = st.text_input("Enter Postcode for Current Location")
    distance_threshold = st.number_input("Maximum Distance for Nearby Stores (in miles)", min_value=1, value=5, step=1)

    # Get number of stores to recommend from user
    max_stores = st.number_input("Number of Nearby Stores to Show", min_value=1, max_value=10, value=3, step=1)

    if postcode:
        current_lat, current_lon = get_coordinates_from_postcode(postcode)
        st.write(current_lat)
        st.write(current_lon)
        if current_lat is not None and current_lon is not None:
            input_data = pd.DataFrame({
                'Store_Size_SQFT': [store_size],
                'Productivity_SQ_F_PerHour': [productivity],
                'Demand_Score': [demand_score],
                'Priority_Level': [priority_level]
            })

            stores_df = pd.read_csv("coop_new_dataset.csv", dtype={"Karcher reference": str})

            if st.button("Predict and Recommend Nearby Stores"):
                predicted_cost = cost_model.predict(input_data)[0]
                predicted_resource = resource_model.predict(input_data)[0]
                resource_type = 'Fixed' if predicted_resource == 1 else 'Mobile'

                st.write("### Prediction Results:")
                st.write(f"**Estimated Service Cost per Visit**: £{predicted_cost:.2f}")
                st.write(f"**Recommended Resource Type**: {resource_type}")

                nearby_stores = get_nearby_stores(current_lat, current_lon, stores_df, distance_threshold, max_stores)
                st.write("### Recommended Nearby Stores:")
                st.dataframe(nearby_stores)

                if not nearby_stores.empty:
                    st.write("### Map with Routes to Nearby Stores:")
                    m = plot_map_with_routes(current_lat, current_lon, nearby_stores)
                    folium_static(m)

            # Optional: If you want to allow users to upload a dataset and apply the model on it
            uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type="csv")
            if uploaded_file:
                # Read the uploaded CSV
                user_data = pd.read_csv(uploaded_file)
                # Apply the same transformations as the input data and make predictions
                user_data['Priority_Level'] = user_data['Priority_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
                user_data['Predicted_Cost'] = cost_model.predict(
                    user_data[["Store_Size_SQFT", "Productivity_SQ_F_PerHour", "Demand_Score", "Priority_Level"]])
                user_data['Predicted_Resource_Type'] = resource_model.predict(
                    user_data[["Store_Size_SQFT", "Productivity_SQ_F_PerHour", "Demand_Score", "Priority_Level"]])
                user_data['Predicted_Resource_Type'] = user_data['Predicted_Resource_Type'].map(
                    {0: 'Mobile', 1: 'Fixed'})

                st.write("### Batch Prediction Results:")
                st.dataframe(
                    user_data[['Store_Size_SQFT', 'Productivity_SQ_F_PerHour', 'Demand_Score', 'Priority_Level',
                               'Predicted_Cost', 'Predicted_Resource_Type']])
