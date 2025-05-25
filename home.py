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
#Chatbot Component
from getpass import getpass
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.messages import AIMessage, HumanMessage
from audio_recorder_streamlit import audio_recorder

#Deployment Key
#openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]
# Read The Data
data = pd.read_csv('./combined_data.csv')
data_viz = pd.read_csv('./lot_size.csv')
df = pd.read_csv('./completed_data_stratxcel.csv')
# Set Streamlit layout to wide
st.set_page_config(layout="wide", page_title="StrataXcel15", page_icon="./building.ico")
#Menu
menu = option_menu(None, ["Home","Dashboard", "Mapping Strata","Chatbot"], 
    icons=['house', 'bar-chart-steps',"globe","robot"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important"},
        "icon": {"color": "white", "font-size": "15px"}, 
        "nav-link": {"font-size": "15   px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
if menu == "Home":
    st.header("Hello, client 😃")
    st.subheader("This project is presented by :blue[Group 15] ", divider="gray")
    st.subheader("Bramanti Brillianto", divider="blue")
    st.subheader("Darmawan Sidiq", divider="green")
    st.subheader("Ibnu Pujiono", divider="orange")
    st.subheader("M. Faisal Zulmy", divider="red")
if menu == "Dashboard":
    def create_sankey_diagram(filtered_df):
        """Create Sankey diagram for filtered data"""
        if filtered_df.empty:
            return None
        
        # Create unique nodes
        all_nodes = []
        node_colors = []
        
        # Add strata_scale nodes
        strata_nodes = filtered_df['strata_scale'].unique()
        all_nodes.extend(strata_nodes)
        node_colors.extend(['#E74C3C'] * len(strata_nodes))
        
        # Add councilarea nodes
        council_nodes = filtered_df['councilarea'].unique()
        all_nodes.extend(council_nodes)
        node_colors.extend(['#3498DB'] * len(council_nodes))
        
        # Add addresssuburb nodes
        suburb_nodes = filtered_df['addresssuburb'].unique()
        all_nodes.extend(suburb_nodes)
        node_colors.extend(['#2ECC71'] * len(suburb_nodes))
        
        # Create node index mapping
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create links
        source = []
        target = []
        value = []
        link_colors = []
        
        # Flow 1: strata_scale -> councilarea
        flow1 = filtered_df.groupby(['strata_scale', 'councilarea']).size().reset_index(name='count')
        for _, row in flow1.iterrows():
            source.append(node_dict[row['strata_scale']])
            target.append(node_dict[row['councilarea']])
            value.append(row['count'])
            link_colors.append('rgba(231, 76, 60, 0.4)')
        
        # Flow 2: councilarea -> addresssuburb
        flow2 = filtered_df.groupby(['councilarea', 'addresssuburb']).size().reset_index(name='count')
        for _, row in flow2.iterrows():
            source.append(node_dict[row['councilarea']])
            target.append(node_dict[row['addresssuburb']])
            value.append(row['count'])
            link_colors.append('rgba(52, 152, 219, 0.4)')
        
        # Create positions
        n_strata = len(strata_nodes)
        n_council = len(council_nodes)
        n_suburb = len(suburb_nodes)
        
        x_positions = []
        y_positions = []
        
        # Strata scale positions (left)
        for i in range(n_strata):
            x_positions.append(0.1)
            y_positions.append((i + 0.5) / max(n_strata, 1))
        
        # Council area positions (middle)
        for i in range(n_council):
            x_positions.append(0.5)
            y_positions.append((i + 0.5) / max(n_council, 1))
        
        # Suburb positions (right)
        for i in range(n_suburb):
            x_positions.append(0.9)
            y_positions.append((i + 0.5) / max(n_suburb, 1))
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 20,
                thickness = 30,
                line = dict(color = "white", width = 2),
                label = all_nodes,
                color = node_colors,
                x = x_positions,
                y = y_positions
            ),
            link = dict(
                source = source,
                target = target,
                value = value,
                color = link_colors,
                line = dict(color = "rgba(0,0,0,0.3)", width = 1)
            )
        )])
        
        fig.update_layout(
            title={
                'text': f"Property Flow Analysis ({len(filtered_df)} properties)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            font=dict(size=12, color='#2C3E50'),
            height=500,
            margin=dict(t=60, b=20, l=20, r=20)
        )
        
        return fig

    def create_summary_tables(filtered_df):
        """Create comprehensive summary tables"""
        if filtered_df.empty:
            return None, None, None, None
        
        try:
            # 1. Council Area Summary
            council_summary = filtered_df.groupby('councilarea').agg({
                'saleslastsoldprice': ['count', 'mean', 'median'],
                'type': lambda x: x.value_counts().to_dict(),
                'strata_scale': lambda x: x.value_counts().to_dict()
            }).round(0)
            
            council_summary.columns = ['Property_Count', 'Avg_Price', 'Median_Value', 'Property_Types', 'Strata_Scales']
            council_summary = council_summary.reset_index()
            
            # 2. Suburb Summary
            suburb_summary = filtered_df.groupby(['councilarea', 'addresssuburb']).agg({
                'saleslastsoldprice': ['count', 'mean', 'median'],
                'type': lambda x: ', '.join(x.value_counts().index[:3]) if len(x.value_counts()) > 0 else 'N/A'
            }).round(0)
            
            suburb_summary.columns = ['Property_Count', 'Avg_Price', 'Median_Value', 'Main_Types']
            suburb_summary = suburb_summary.reset_index()
            
            # 3. Property Type Summary
            type_summary = filtered_df.groupby(['councilarea', 'type']).agg({
                'saleslastsoldprice': ['count', 'mean'],
                'strata_scale': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
            }).round(0)
            
            type_summary.columns = ['Count', 'Avg_Price', 'Common_Scale']
            type_summary = type_summary.reset_index()
            
            # 4. Strata Scale Summary
            strata_summary = filtered_df.groupby(['councilarea', 'strata_scale']).agg({
                'saleslastsoldprice': ['count', 'mean', 'median', 'max']
            }).round(0)
            
            strata_summary.columns = ['Count', 'Avg_Price', 'Median_Price', 'Max_Price']
            strata_summary = strata_summary.reset_index()
            
            return council_summary, suburb_summary, type_summary, strata_summary
            
        except Exception as e:
            st.error(f"Error creating summary tables: {str(e)}")
            return None, None, None, None

    st.title("🏢 Strata Building Analysis Dashboard")
    st.markdown("### Council Area Analysis")
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Get unique council areas
    unique_councils = sorted(df['councilarea'].unique())
    
    # Multiselect for council areas with "All" option
    st.sidebar.markdown("**Select Council Areas:**")
    
    # Add "All" checkbox
    select_all = st.sidebar.checkbox("Select All Council Areas", value=True)
    
    if select_all:
        selected_councils = st.sidebar.multiselect(
            "Council Areas",
            options=unique_councils,
            default=unique_councils,
            disabled=True
        )
    else:
        selected_councils = st.sidebar.multiselect(
            "Council Areas",
            options=unique_councils,
            default=unique_councils[:2] if len(unique_councils) >= 2 else unique_councils
        )
    
    # Additional filters
    st.sidebar.markdown("**Additional Filters:**")
    
    # Price range filter
    min_price, max_price = st.sidebar.slider(
        "Price Range ($)",
        min_value=int(df['saleslastsoldprice'].min()),
        max_value=int(df['saleslastsoldprice'].max()),
        value=(int(df['saleslastsoldprice'].min()), int(df['saleslastsoldprice'].max())),
        step=50000,
        format="$%d"
    )
    
    # Property type filter
    selected_types = st.sidebar.multiselect(
        "Property Types",
        options=sorted(df['type'].unique()),
        default=sorted(df['type'].unique())
    )
    
    # Filter the dataframe
    if selected_councils:
        filtered_df = df[
            (df['councilarea'].isin(selected_councils)) &
            (df['saleslastsoldprice'] >= min_price) &
            (df['saleslastsoldprice'] <= max_price) &
            (df['type'].isin(selected_types))
        ]
    else:
        st.warning("⚠️ Please select at least one council area.")
        filtered_df = pd.DataFrame()
    
    if not filtered_df.empty:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Properties", len(filtered_df))
        
        with col2:
            avg_price = filtered_df['saleslastsoldprice'].mean()
            st.metric("Average Price", f"${avg_price:,.0f}")
        
        with col3:
            median_value = filtered_df['saleslastsoldprice'].median()
            st.metric("Median Price", f"${median_value:,.0f}")
        
        with col4:
            unique_suburbs = filtered_df['addresssuburb'].nunique()
            st.metric("Unique Suburbs", unique_suburbs)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary Tables", "🌊 Sankey Diagram", "📈 Charts"])
        
        with tab1:
            st.markdown("## 📊 Summary Tables")
            
            # Get summary tables
            council_summary, suburb_summary, type_summary, strata_summary = create_summary_tables(filtered_df)
            
            # Display summary tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Council Area Summary")
                if council_summary is not None:
                    # Format the summary for better display
                    display_council = council_summary.copy()
                    display_council['Avg_Price'] = display_council['Avg_Price'].apply(lambda x: f"${x:,.0f}")
                    display_council['Median_Value'] = display_council['Median_Value'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(display_council[['councilarea', 'Property_Count', 'Avg_Price', 'Median_Value']], 
                               hide_index=True, use_container_width=True)
                
                st.markdown("### Property Type Summary")
                if type_summary is not None:
                    display_type = type_summary.copy()
                    display_type['Avg_Price'] = display_type['Avg_Price'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(display_type, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("### Suburb Summary")
                if suburb_summary is not None:
                    display_suburb = suburb_summary.copy()
                    display_suburb['Avg_Price'] = display_suburb['Avg_Price'].apply(lambda x: f"${x:,.0f}")
                    display_suburb['Median_Value'] = display_suburb['Median_Value'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(display_suburb, hide_index=True, use_container_width=True)
                
                st.markdown("### Strata Scale Summary")
                if strata_summary is not None:
                    display_strata = strata_summary.copy()
                    for col in ['Avg_Price', 'Median_Price', 'Max_Price']:
                        display_strata[col] = display_strata[col].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(display_strata, hide_index=True, use_container_width=True)
        
        with tab2:
            st.markdown("## 🌊 Sankey Diagram")
            sankey_fig = create_sankey_diagram(filtered_df)
            if sankey_fig:
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")
        
        with tab3:
            st.markdown("## 📈 Charts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution by council area
                fig_box = px.box(filtered_df, x='councilarea', y='saleslastsoldprice', 
                               title='Price Distribution by Council Area')
                fig_box.update_layout(xaxis_title='Council Area', yaxis_title='Sale Price ($)')
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Property count by type
                type_counts = filtered_df['type'].value_counts()
                fig_pie = px.pie(values=type_counts.values, names=type_counts.index, 
                               title='Property Distribution by Type')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Strata scale distribution
            strata_counts = filtered_df.groupby(['councilarea', 'strata_scale']).size().reset_index(name='count')
            fig_bar = px.bar(strata_counts, x='councilarea', y='count', color='strata_scale',
                           title='Strata Scale Distribution by Council Area')
            st.plotly_chart(fig_bar, use_container_width=True)
         
    else:
        st.info("🔍 No properties match the selected criteria. Please adjust your filters.")
    

# if menu == "Descriptive Statistic":
#     st.title("Descriptive Statistics")
#     st.write(data.head())
#     location = st.selectbox( "Choose your bulding location...",data["Address Complete"].unique())
#     st.write("Your current location is", location)
#     with st.expander("Building Data"):
#         col1, col2 = st.columns(2)
#     with col1:
#         st.metric(label="Suburb",value=data.loc[data["Address Complete"] == location, "Suburb"].iloc[0])
#         st.metric(label="Year Built",value=int(data.loc[data["Address Complete"] == location, "Year Built"].iloc[0]))
#         st.metric(label="Number of Bedrooms",value=int(data.loc[data["Address Complete"] == location, "Beds"].iloc[0]))
#         st.metric(label="Occupancy Type",value=str(data.loc[data["Address Complete"] == location, "Occupancy Type"].iloc[0]))
#     with col2:
#          st.metric(label="Property Type",value=data.loc[data["Address Complete"] == location, "Property Type"].iloc[0])
#          st.metric(label="Land Area",value=str(data.loc[data["Address Complete"] == location, "Land Area"].iloc[0]) + " m2")
#          st.metric(label="Number of Bathrooms",value=int(data.loc[data["Address Complete"] == location, "Baths"].iloc[0]))
#          st.metric(label="Price", value=int(data.loc[data["Address Complete"] == location, "Sales Price"].iloc[0]))

#     st.write("Location to nearest public places...")
#     # Define location
#     place_name = "Brunswick, Victoria, Australia"

#     # Download the street network
#     G = ox.graph_from_place(place_name, network_type="walk")  # Use "drive" for car routes

#     # Define the starting coordinate (Example: A central point in Brunswick)
#     start_lat = data.loc[data["Address Complete"] == location, "Lat"].iloc[0]
#     start_lon = data.loc[data["Address Complete"] == location, "Long"].iloc[0]

#     # Find nearest node in the network to the starting coordinate
#     orig_node = ox.nearest_nodes(G, start_lon, start_lat)

#     # Overpass Query for Public Places
#     overpass_url = "http://overpass-api.de/api/interpreter"
#     overpass_query = """
#     [out:json];
#     area[name="Brunswick"]->.searchArea;

#     (
#     node["shop"="supermarket"](area.searchArea);
#     node["amenity"="bus_station"](area.searchArea);
#     node["railway"="station"](area.searchArea);
#     node["amenity"="hospital"](area.searchArea);
#     node["amenity"="clinic"](area.searchArea);
#     node["amenity"="school"](area.searchArea);
#     node["amenity"="university"](area.searchArea);
#     node["leisure"="park"](area.searchArea);
#     node["amenity"="police"](area.searchArea);
#     node["amenity"="fire_station"](area.searchArea);
#     node["amenity"="post_office"](area.searchArea);
#     );
#     out center;
#     """

#     # Fetch Public Places Data
#     response = requests.get(overpass_url, params={'data': overpass_query})
#     data = response.json()

#     # Extract public places and find nearest nodes
#     locations = []
#     dest_nodes = []
#     distances = {}

#     for element in data["elements"]:
#         if "lat" in element and "lon" in element:
#             place_lat, place_lon = element["lat"], element["lon"]
#             place_type = list(element["tags"].values())[0]  # Get type
#             nearest_node = ox.nearest_nodes(G, place_lon, place_lat)  # Find nearest network node
            
#             try:
#                 # Compute shortest path distance
#                 distance = nx.shortest_path_length(G, orig_node, nearest_node, weight="length")
#                 distances[nearest_node] = distance  # Store distance for marker color
#             except nx.NetworkXNoPath:
#                 distance = None

#             locations.append({
#                 "name": element["tags"].get("name", "Unknown"),
#                 "lat": place_lat,
#                 "lon": place_lon,
#                 "type": place_type,
#                 "node": nearest_node,
#                 "distance": distance  # Distance in meters
#             })
#             dest_nodes.append(nearest_node)

#     # Define color gradient based on distance
#     def get_color(distance):
#         if distance is None:
#             return "gray"
#         elif distance < 1000:
#             return "green"
#         elif distance < 3000:
#             return "orange"
#         else:
#             return "red"

#     # Create Folium Map
#     m = folium.Map(location=[start_lat, start_lon], zoom_start=14)

#     # Add Origin Marker
#     folium.Marker([start_lat, start_lon], 
#                 popup="Start Location", 
#                 icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

#     # Add Destination Markers with Distance Info
#     marker_cluster = MarkerCluster().add_to(m)

#     for loc in locations:
#         color = get_color(loc["distance"])
#         distance_text = f"{round(loc['distance'] / 1000, 2)} km" if loc["distance"] else "No path"

#         folium.Marker(
#             location=[loc["lat"], loc["lon"]],
#             popup=f"{loc['name']} ({loc['type']})<br>Distance: {distance_text}",
#             icon=folium.Icon(color=color)
#         ).add_to(marker_cluster)

#     # Plot Shortest Paths
#     for loc in locations:
#         if loc["distance"]:  # Only plot paths where a distance was found
#             path = nx.shortest_path(G, orig_node, loc["node"], weight="length")
#             path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
#             folium.PolyLine(path_coords, color="blue", weight=3, opacity=0.7).add_to(m)

#     # Show Map
#     st_folium(m, width=800,height=600)

# if menu == "Clustering":
#     st.title("Clustering")
#     # Number of clusters
#     k = 4

#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     data_viz['Cluster'] = kmeans.fit_predict(data_viz[['lat', 'long']])

#     # Define Colors for Clusters
#     colors = ['red', 'blue', 'green', 'purple', 'orange']
#     data_viz['Color'] = data_viz['Cluster'].apply(lambda x: colors[x])

#     # Create a Folium Map
#     m = folium.Map(location=[data_viz['lat'].mean(), data_viz['long'].mean()], zoom_start=14)

#     # Add Points to the Map
#     marker_cluster = MarkerCluster().add_to(m)

#     for idx, row in data_viz.iterrows():
#         folium.CircleMarker(
#             location=[row['lat'], row['long']],
#             radius=7,
#             color=row['Color'],
#             fill=True,
#             fill_color=row['Color'],
#             fill_opacity=0.6,
#             popup=f"Cluster {row['Cluster']}"
#         ).add_to(marker_cluster)

#     # Display Map
#     st_folium(m, width=900,height=600)
if menu == "Mapping Strata":
    def create_folium_map(df_filtered, highlight_coords=None):
        """Create a Folium map with the filtered data and optional highlight"""
        center_lat = df_filtered['addresslocation_lat'].mean() if len(df_filtered) > 0 else -37.8
        center_lon = df_filtered['addresslocation_lon'].mean() if len(df_filtered) > 0 else 145.0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
        
        color_map = {
            'small scale': '#FF6B6B',
            'medium scale': '#4ECDC4'
        }

        for idx, row in df_filtered.iterrows():
            coords = [row['addresslocation_lat'], row['addresslocation_lon']]
            
            # Check if this point is the one clicked
            if highlight_coords and coords == highlight_coords:
                # Highlighted marker
                folium.CircleMarker(
                    location=coords,
                    radius=12,
                    popup=row['building_id'],
                    color='black',
                    weight=3,
                    fill=True,
                    fill_color='yellow',
                    fill_opacity=1,
                    tooltip="Selected"
                ).add_to(m)
            
            # Regular marker
            folium.CircleMarker(
                location=coords,
                radius=8,
                popup=row['building_id'],
                color='white',
                weight=2,
                fill=True,
                fill_color=color_map.get(row['strata_scale'], '#999999'),
                fill_opacity=0.8,
                tooltip=row['addresscomplete']
            ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 80px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Strata Scale</b></p>
        <p><i class="fa fa-circle" style="color:#4ECDC4"></i> Medium Scale</p>
        <p><i class="fa fa-circle" style="color:#FF6B6B"></i> Small Scale</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m

    st.title("🏢 Strata Properties Map Filter")
    st.sidebar.header("🔍 Filters")
    available_scales = sorted(df['strata_scale'].dropna().unique())
    selected_scales = st.sidebar.multiselect("Select Strata Scale(s):", options=available_scales, default=available_scales)
    
    min_price = int(df['saleslastsoldprice'].min())
    max_price = int(df['saleslastsoldprice'].max())
    price_range = st.sidebar.slider("Price Range ($):", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=10000)
    
    council_areas = sorted(df['councilarea'].dropna().unique())
    selected_councils = st.sidebar.multiselect("Council Area(s):", options=council_areas, default="MARIBYRNONG")
    
    df_filtered = df[
        (df['strata_scale'].isin(selected_scales)) &
        (df['saleslastsoldprice'] >= price_range[0]) &
        (df['saleslastsoldprice'] <= price_range[1]) &
        (df['councilarea'].isin(selected_councils))
    ]
    
    st.subheader("📍 Interactive Map")

    # Initial map without any selected point
    folium_map = create_folium_map(df_filtered)
    map_data = st_folium(folium_map, width=700, height=500, returned_objects=["last_clicked"])

    # Check for click
    clicked_df = df_filtered
    highlight_coords = None

    if map_data and map_data["last_clicked"]:
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        highlight_coords = [lat, lon]
        
        matched_row = df_filtered[
            (df_filtered['addresslocation_lat'].round(6) == round(lat, 6)) &
            (df_filtered['addresslocation_lon'].round(6) == round(lon, 6))
        ]
        
        if not matched_row.empty:
            building_id = matched_row.iloc[0]['building_id']
            clicked_df = df[df['building_id'] == building_id]
    
    # Re-render map with highlight
    folium_map = create_folium_map(df_filtered, highlight_coords)
    st_folium(folium_map, width=700, height=500)

    # Display clicked or filtered data
    st.subheader("📋 Property Details")
    display_columns = ['addresscomplete', 'strata_scale', 'saleslastsoldprice', 'councilarea', 'building_id', 'unit_number']
    st.dataframe(
        clicked_df[display_columns].rename(columns={
            'addresscomplete': 'Address',
            'strata_scale': 'Strata Scale',
            'saleslastsoldprice': 'Last Sold Price',
            'councilarea': 'Council Area',
            'building_id': 'Building ID',
            'unit_number': 'Unit Number'
        }),
        use_container_width=True
    )

if menu == "Chatbot":
    llm = ChatGroq(temperature=0, model="llama-3.2-90b-text-preview", api_key=groq_api_key)
    #llm = ChatOpenAI(model="gpt-4-turbo", temperature=0,api_key=openai_api_key)
    agent = create_csv_agent(llm, "./completed_data_stratxcel.csv", verbose=True, allow_dangerous_code=True, max_execution_time=1000000000000000)

    # Toggle for input mode: Text or Speech
    
    input_mode = st.selectbox("Choose your input method:", ("Text", "Speech"))
    user_query = None
    if input_mode == "Text":
        user_query = st.chat_input("Type your message here...")
        if "context_log" not in st.session_state:
            st.session_state.context_log = ["Retrieved context will be displayed here"]
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hi, I'm a friendly assistant. How can I help you?")]
        result = st.toggle("Toggle Context")
        if result:
            st.write(st.session_state.context_log)


        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
            
            with st.chat_message("AI"):
                response = st.write_stream(get_response(user_query))
            
            st.session_state.chat_history.append(AIMessage(content=response))
    else:
        st.write("Click the button to record your speech:")
        audio_bytes = audio_recorder()  # Record audio using audio-recorder-streamlit
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")  # Playback the recorded audio
            # Process audio for speech-to-text (if needed, using an external API like Google or OpenAI Whisper)
            # Example: Convert audio to text using Google Speech Recognition or any other service.
            try:
                from io import BytesIO
                import speech_recognition as sr

                recognizer = sr.Recognizer()
                with sr.AudioFile(BytesIO(audio_bytes)) as source:
                    audio = recognizer.record(source)
                    user_query = recognizer.recognize_google(audio)  # Convert speech to text
                    st.write("Recognized Text:", user_query)

            except Exception as e:
                st.write("Could not process audio:", str(e))

        if user_query is not None and user_query != "":
            st.chat_message('user').markdown(user_query)
            st.session_state.messages.append({'role': 'user', 'content': user_query})
            with st.spinner("Thinking...and Calculating"):
                response = agent.invoke(user_query)
            # Display the assistant's response
            st.chat_message('assistant').markdown(response['output'])
            st.session_state.messages.append({'role': 'assistant', 'content': response['output']})
# Footer
st.markdown("---")
st.markdown("Built with love and passion ❤‍🔥 from Group 15")