import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from folium.plugins import MarkerCluster
from datetime import date
import plotly.express as px
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
import time
import osmnx as ox
import networkx as nx
import folium
import requests
from folium.plugins import MarkerCluster
import geopandas as gpd
import folium
from geopy.geocoders import Nominatim
import requests
from shapely.geometry import Polygon
from streamlit_folium import st_folium

# Read The Data
data = pd.read_csv('./combined_data.csv')
# Set Streamlit layout to wide
st.set_page_config(layout="wide", page_title="StrataXcel15", page_icon="./building.ico")
#Menu
menu = option_menu(None, ["Home","Descriptive Statistic","Clustering"], 
    icons=['house', 'bar-chart-steps','buildings'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important"},
        "icon": {"color": "white", "font-size": "15px"}, 
        "nav-link": {"font-size": "15   px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
if menu == "Home":
    st.header("Hello, user 😃")
    st.subheader("This project is presented by :blue[Group 15] ", divider="gray")
    st.subheader("Bramanti Brillianto", divider="blue")
    st.subheader("Darmawan Sidiq", divider="green")
    st.subheader("Ibnu Pujiono", divider="orange")
    st.subheader("M. Faisal Zulmy", divider="red")
if menu == "Descriptive Statistic":
    st.title("Descriptive Statistics")
    st.write(data.head())
    location = st.selectbox( "Choose your bulding location...",data["Address Complete"].unique())
    st.write("Your current location is", location)
    with st.expander("Building Data"):
        col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Suburb",value=data.loc[data["Address Complete"] == location, "Suburb"].iloc[0])
        st.metric(label="Year Built",value=int(data.loc[data["Address Complete"] == location, "Year Built"].iloc[0]))
        st.metric(label="Number of Bedrooms",value=int(data.loc[data["Address Complete"] == location, "Beds"].iloc[0]))
        st.metric(label="Occupancy Type",value=str(data.loc[data["Address Complete"] == location, "Occupancy Type"].iloc[0]))
    with col2:
         st.metric(label="Property Type",value=data.loc[data["Address Complete"] == location, "Property Type"].iloc[0])
         st.metric(label="Land Area",value=str(data.loc[data["Address Complete"] == location, "Land Area"].iloc[0]) + " m2")
         st.metric(label="Number of Bathrooms",value=int(data.loc[data["Address Complete"] == location, "Baths"].iloc[0]))
         st.metric(label="Price", value=int(data.loc[data["Address Complete"] == location, "Sales Price"].iloc[0]))

    st.write("Location to nearest public places...")
    # Define location
    place_name = "Brunswick, Victoria, Australia"

    # Download the street network
    G = ox.graph_from_place(place_name, network_type="walk")  # Use "drive" for car routes

    # Define the starting coordinate (Example: A central point in Brunswick)
    start_lat = data.loc[data["Address Complete"] == location, "Lat"].iloc[0]
    start_lon = data.loc[data["Address Complete"] == location, "Long"].iloc[0]

    # Find nearest node in the network to the starting coordinate
    orig_node = ox.nearest_nodes(G, start_lon, start_lat)

    # Overpass Query for Public Places
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    area[name="Brunswick"]->.searchArea;

    (
    node["shop"="supermarket"](area.searchArea);
    node["amenity"="bus_station"](area.searchArea);
    node["railway"="station"](area.searchArea);
    node["amenity"="hospital"](area.searchArea);
    node["amenity"="clinic"](area.searchArea);
    node["amenity"="school"](area.searchArea);
    node["amenity"="university"](area.searchArea);
    node["leisure"="park"](area.searchArea);
    node["amenity"="police"](area.searchArea);
    node["amenity"="fire_station"](area.searchArea);
    node["amenity"="post_office"](area.searchArea);
    );
    out center;
    """

    # Fetch Public Places Data
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    # Extract public places and find nearest nodes
    locations = []
    dest_nodes = []
    distances = {}

    for element in data["elements"]:
        if "lat" in element and "lon" in element:
            place_lat, place_lon = element["lat"], element["lon"]
            place_type = list(element["tags"].values())[0]  # Get type
            nearest_node = ox.nearest_nodes(G, place_lon, place_lat)  # Find nearest network node
            
            try:
                # Compute shortest path distance
                distance = nx.shortest_path_length(G, orig_node, nearest_node, weight="length")
                distances[nearest_node] = distance  # Store distance for marker color
            except nx.NetworkXNoPath:
                distance = None

            locations.append({
                "name": element["tags"].get("name", "Unknown"),
                "lat": place_lat,
                "lon": place_lon,
                "type": place_type,
                "node": nearest_node,
                "distance": distance  # Distance in meters
            })
            dest_nodes.append(nearest_node)

    # Define color gradient based on distance
    def get_color(distance):
        if distance is None:
            return "gray"
        elif distance < 1000:
            return "green"
        elif distance < 3000:
            return "orange"
        else:
            return "red"

    # Create Folium Map
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14)

    # Add Origin Marker
    folium.Marker([start_lat, start_lon], 
                popup="Start Location", 
                icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

    # Add Destination Markers with Distance Info
    marker_cluster = MarkerCluster().add_to(m)

    for loc in locations:
        color = get_color(loc["distance"])
        distance_text = f"{round(loc['distance'] / 1000, 2)} km" if loc["distance"] else "No path"

        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"{loc['name']} ({loc['type']})<br>Distance: {distance_text}",
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)

    # Plot Shortest Paths
    for loc in locations:
        if loc["distance"]:  # Only plot paths where a distance was found
            path = nx.shortest_path(G, orig_node, loc["node"], weight="length")
            path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
            folium.PolyLine(path_coords, color="blue", weight=3, opacity=0.7).add_to(m)

    # Show Map
    st_folium(m, width=800,height=600)

if menu == "Clustering":
    st.title("Clustering")
    # Number of clusters
    k = 4

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(data[['Lat', 'Long']])

    # Define Colors for Clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    data['Color'] = data['Cluster'].apply(lambda x: colors[x])

    # Create a Folium Map
    m = folium.Map(location=[data['Lat'].mean(), data['Long'].mean()], zoom_start=14)

    # Add Points to the Map
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Long']],
            radius=7,
            color=row['Color'],
            fill=True,
            fill_color=row['Color'],
            fill_opacity=0.6,
            popup=f"Cluster {row['Cluster']}"
        ).add_to(marker_cluster)

    # Display Map
    st_folium(m, width=900,height=600)