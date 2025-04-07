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
data_viz = pd.read_csv('./lot_size.csv')
# Set Streamlit layout to wide
st.set_page_config(layout="wide", page_title="StrataXcel15", page_icon="./building.ico")
#Menu
menu = option_menu(None, ["Home","Dashboard","Descriptive Statistic","Clustering"], 
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
if menu == "Dashboard":
    st.header("Dashboard")
    grouped_sorted = data_viz
    #Sankey Viz
    import plotly.graph_objects as go

    # Create all nodes: combine council_area, zone, property_type, lot_size
    all_nodes = list(pd.unique(grouped_sorted['council_area'].tolist() +
                                grouped_sorted['development_zone_base'].tolist() +
                                grouped_sorted['property_type'].tolist() +
                                grouped_sorted['lot_size'].tolist()))
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    # Generate colors for the nodes
    council_colors = ['#FF5733', '#FFC300', '#DAF7A6', '#C70039', '#900C3F', '#581845', '#2980B9', '#8E44AD']
    zone_colors = ['#1F618D', '#D35400', '#7D3C98', '#27AE60', '#E74C3C', '#F39C12']
    property_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F1C40F', '#E67E22', '#1ABC9C']
    lot_size_colors = {'Large': '#2E86C1', 'Medium': '#58D68D', 'Small': '#F7DC6F'}

    # Create links for council_area → zone
    links_council_zone = grouped_sorted.groupby(['council_area', 'development_zone_base'])['property_count'].sum().reset_index()
    links_council_zone['source'] = links_council_zone['council_area'].map(node_indices)
    links_council_zone['target'] = links_council_zone['development_zone_base'].map(node_indices)

    # Create links for zone → property_type
    links_zone_type = grouped_sorted.groupby(['development_zone_base', 'property_type'])['property_count'].sum().reset_index()
    links_zone_type['source'] = links_zone_type['development_zone_base'].map(node_indices)
    links_zone_type['target'] = links_zone_type['property_type'].map(node_indices)

    # Create links for property_type → lot_size
    links_type_lot = grouped_sorted.groupby(['property_type', 'lot_size'])['property_count'].sum().reset_index()
    links_type_lot['source'] = links_type_lot['property_type'].map(node_indices)
    links_type_lot['target'] = links_type_lot['lot_size'].map(node_indices)

    # Combine all link sets
    sources = links_council_zone['source'].tolist() + links_zone_type['source'].tolist() + links_type_lot['source'].tolist()
    targets = links_council_zone['target'].tolist() + links_zone_type['target'].tolist() + links_type_lot['target'].tolist()
    values = links_council_zone['property_count'].tolist() + links_zone_type['property_count'].tolist() + links_type_lot['property_count'].tolist()

    # Create a color array for the links: alternating between different zones and property types
    link_colors = []
    for idx, (source, target) in enumerate(zip(sources, targets)):
        if all_nodes[source] in council_colors:
            link_colors.append('#1F618D')  # Color for council_area to zone links
        elif all_nodes[source] in zone_colors:
            link_colors.append('#E74C3C')  # Color for zone to property type links
        else:
            link_colors.append('#F7DC6F')  # Color for property_type to lot_size links

    # Build the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=[council_colors[i % len(council_colors)] if all_nodes[i] in council_colors
                else zone_colors[i % len(zone_colors)] if all_nodes[i] in zone_colors
                else lot_size_colors.get(all_nodes[i], '#9B59B6') for i in range(len(all_nodes))]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(title_text="🏗️ Property Categorization: Council Area → Zone → Property Type → Lot Size", font_size=12)
    st.plotly_chart(fig)
    council_select = st.sidebar.selectbox("Choose Council Area",grouped_sorted["council_area"].unique())
    st.sidebar.write("Council Area : ",council_select)

    #Filter Data
    data_filter = grouped_sorted[grouped_sorted["council_area"] == council_select]
    fig = px.pie(data_filter, values='property_count', names='development_zone_base', title='Property Development Area in ' + council_select)
    st.plotly_chart(fig)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(data_filter, values='property_count', names='suburb', title='Suburb location in ' + council_select)
        st.plotly_chart(fig)
    with col2:
        fig = px.pie(data_filter, values='property_count', names='lot_size', title='Lot Size in ' + council_select)
        st.plotly_chart(fig)
    
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