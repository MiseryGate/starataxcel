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
import speech_recognition as sr
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
    
if menu == "Mapping Strata":
    def create_heatmap_data(df, scale_type):
        """Prepare data for heatmap visualization"""
        scale_data = df[df['strata_scale'] == scale_type].copy()
        
        # Create heatmap points with weights (can be count, price, or other metrics)
        heatmap_data = []
        for idx, row in scale_data.iterrows():
            heatmap_data.append([
                row['addresslocation_lat'],
                row['addresslocation_lon'],
                1  # Weight - you can modify this to use price or other metrics
            ])
        
        return heatmap_data

    def create_heatmap(df_filtered, selected_scales):
        """Create a heatmap showing distribution of strata scales"""
        from folium.plugins import HeatMap
        
        center_lat = df_filtered['addresslocation_lat'].mean() if len(df_filtered) > 0 else -37.8
        center_lon = df_filtered['addresslocation_lon'].mean() if len(df_filtered) > 0 else 145.0
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=12, 
            tiles='OpenStreetMap'
        )
        
        # Color scheme for different scales
        colors = {
            'small scale': '#FF6B6B',
            'medium scale': '#4ECDC4'
        }
        
        # Add heatmaps for each selected scale
        for scale in selected_scales:
            if scale in df_filtered['strata_scale'].values:
                heatmap_data = create_heatmap_data(df_filtered, scale)
                
                if heatmap_data:
                    HeatMap(
                        heatmap_data,
                        min_opacity=0.3,
                        max_zoom=18,
                        radius=25,
                        blur=15,
                        gradient={
                            0.4: colors.get(scale, '#999999'),
                            0.6: colors.get(scale, '#999999'),
                            0.8: colors.get(scale, '#999999'),
                            1.0: colors.get(scale, '#999999')
                        }
                    ).add_to(m)
        
        # Add density markers for reference
        building_density = df_filtered.groupby(['addresslocation_lat', 'addresslocation_lon']).agg({
            'building_id': 'nunique',
            'strata_scale': lambda x: ', '.join(x.unique()),
            'addresscomplete': 'first'
        }).reset_index()
        
        for idx, row in building_density.iterrows():
            if row['building_id'] > 1:  # Only show areas with multiple buildings
                folium.CircleMarker(
                    location=[row['addresslocation_lat'], row['addresslocation_lon']],
                    radius=min(row['building_id'] * 2, 15),
                    popup=f"Buildings: {row['building_id']}<br>Scales: {row['strata_scale']}<br>Area: {row['addresscomplete']}",
                    color='white',
                    weight=2,
                    fill=True,
                    fill_color='orange',
                    fill_opacity=0.7,
                    tooltip=f"Density: {row['building_id']} buildings"
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Strata Heatmap Legend</b></p>
        <p><span style="color:#FF6B6B">●</span> Small Scale Density</p>
        <p><span style="color:#4ECDC4">●</span> Medium Scale Density</p>
        <p><span style="color:orange">●</span> High Building Density Areas</p>
        <p><em>Intensity shows concentration</em></p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m

    def create_comparison_heatmap(df_filtered):
        """Create side-by-side heatmap comparison"""
        from folium.plugins import HeatMap
        
        center_lat = df_filtered['addresslocation_lat'].mean() if len(df_filtered) > 0 else -37.8
        center_lon = df_filtered['addresslocation_lon'].mean() if len(df_filtered) > 0 else 145.0
        
        # Create comparison data
        small_scale_data = create_heatmap_data(df_filtered, 'small scale')
        medium_scale_data = create_heatmap_data(df_filtered, 'medium scale')
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=12, 
            tiles='OpenStreetMap'
        )
        
        # Add small scale heatmap
        if small_scale_data:
            HeatMap(
                small_scale_data,
                min_opacity=0.2,
                max_zoom=18,
                radius=20,
                blur=12,
                gradient={0.4: '#FFE5E5', 0.6: '#FF9999', 0.8: '#FF6B6B', 1.0: '#FF4444'}
            ).add_to(m)
        
        # Add medium scale heatmap  
        if medium_scale_data:
            HeatMap(
                medium_scale_data,
                min_opacity=0.2,
                max_zoom=18,
                radius=20,
                blur=12,
                gradient={0.4: '#E5F9F6', 0.6: '#99E6D9', 0.8: '#4ECDC4', 1.0: '#44C4B8'}
            ).add_to(m)
        
        # Add statistical markers
        stats_data = df_filtered.groupby(['councilarea', 'strata_scale']).agg({
            'building_id': 'nunique',
            'addresslocation_lat': 'mean',
            'addresslocation_lon': 'mean'
        }).reset_index()
        
        for idx, row in stats_data.iterrows():
            color = '#FF6B6B' if row['strata_scale'] == 'small scale' else '#4ECDC4'
            folium.Marker(
                location=[row['addresslocation_lat'], row['addresslocation_lon']],
                popup=f"Council: {row['councilarea']}<br>Scale: {row['strata_scale']}<br>Buildings: {row['building_id']}",
                tooltip=f"{row['councilarea']}: {row['building_id']} {row['strata_scale']} buildings",
                icon=folium.Icon(color='red' if row['strata_scale'] == 'small scale' else 'blue', icon='building', prefix='fa')
            ).add_to(m)
        
        return m

    def create_spatial_index(df):
        """Create a spatial index for fast coordinate lookups"""
        if 'spatial_index' not in st.session_state or st.session_state.get('spatial_index_df_id') != id(df):
            # Create a dictionary for O(1) lookups
            spatial_dict = {}
            for idx, row in df.iterrows():
                lat_rounded = round(row['addresslocation_lat'], 6)
                lon_rounded = round(row['addresslocation_lon'], 6)
                key = (lat_rounded, lon_rounded)
                
                if key not in spatial_dict:
                    spatial_dict[key] = []
                spatial_dict[key].append({
                    'building_id': row['building_id'],
                    'index': idx
                })
            
            st.session_state.spatial_index = spatial_dict
            st.session_state.spatial_index_df_id = id(df)
        
        return st.session_state.spatial_index

    def find_building_from_coords(lat, lon, spatial_index, tolerance=0.0001):
        """Fast coordinate-based building lookup with tolerance"""
        lat_rounded = round(lat, 6)
        lon_rounded = round(lon, 6)
        
        # Try exact match first
        key = (lat_rounded, lon_rounded)
        if key in spatial_index:
            return spatial_index[key][0]['building_id']
        
        # Try nearby coordinates within tolerance
        for coord_key in spatial_index.keys():
            if (abs(coord_key[0] - lat_rounded) <= tolerance and 
                abs(coord_key[1] - lon_rounded) <= tolerance):
                return spatial_index[coord_key][0]['building_id']
        
        return None

    def get_building_data(building_id, df):
        """Get all strata data for a specific building"""
        if building_id is None:
            return pd.DataFrame()
        
        # Filter data for the specific building
        building_data = df[df['building_id'] == building_id].copy()
        
        # Sort by unit number if available
        if 'unit_number' in building_data.columns:
            building_data = building_data.sort_values('unit_number')
        
        return building_data

    def create_building_summary(building_data):
        """Create a summary of the building's strata data"""
        if building_data.empty:
            return {}
        
        summary = {
            'Total Units': len(building_data),
            'Building Address': building_data.iloc[0]['addresscomplete'],
            'Council Area': building_data.iloc[0]['councilarea'],
            'Strata Scale': building_data.iloc[0]['strata_scale'],
            'Average Sale Price': f"${building_data['saleslastsoldprice'].mean():,.0f}",
            'Price Range': f"${building_data['saleslastsoldprice'].min():,.0f} - ${building_data['saleslastsoldprice'].max():,.0f}",
            'Property Types': ', '.join(building_data['type'].unique()),
            'Latest Sale Date': building_data['saleslastsolddate'].max() if 'saleslastsolddate' in building_data.columns else 'N/A'
        }
        
        return summary

    def create_folium_map(df_filtered, highlight_building_id=None):
        """Create a Folium map with the filtered data and optional building highlight"""
        if len(df_filtered) == 0:
            center_lat, center_lon = -37.8, 145.0
        else:
            center_lat = df_filtered['addresslocation_lat'].mean()
            center_lon = df_filtered['addresslocation_lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
        
        color_map = {
            'small scale': '#FF6B6B',
            'medium scale': '#4ECDC4'
        }

        # Group by building_id to avoid duplicate markers
        building_groups = df_filtered.groupby('building_id').first().reset_index()
        
        for idx, row in building_groups.iterrows():
            coords = [row['addresslocation_lat'], row['addresslocation_lon']]
            building_id = row['building_id']
            
            # Count units in this building
            unit_count = len(df_filtered[df_filtered['building_id'] == building_id])
            
            # Determine if this building should be highlighted
            is_highlighted = (highlight_building_id == building_id)
            
            # Create popup content with building summary
            popup_content = f"""
            <div style="width: 250px;">
                <h4>{row['addresscomplete']}</h4>
                <p><strong>Building ID:</strong> {building_id}</p>
                <p><strong>Units:</strong> {unit_count}</p>
                <p><strong>Strata Scale:</strong> {row['strata_scale']}</p>
                <p><strong>Council:</strong> {row['councilarea']}</p>
                <p><em>Click to view all units</em></p>
            </div>
            """
            
            # Create marker
            folium.CircleMarker(
                location=coords,
                radius=15 if is_highlighted else 10,
                popup=folium.Popup(popup_content, max_width=300),
                color='black' if is_highlighted else 'white',
                weight=3,
                fill=True,
                fill_color='yellow' if is_highlighted else color_map.get(row['strata_scale'], '#999999'),
                fill_opacity=1 if is_highlighted else 0.8,
                tooltip=f"{row['addresscomplete']} ({unit_count} units)"
            ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 100px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Strata Scale</b></p>
        <p><i class="fa fa-circle" style="color:#4ECDC4"></i> Medium Scale</p>
        <p><i class="fa fa-circle" style="color:#FF6B6B"></i> Small Scale</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Selected Building</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m

    st.title("🏢 Strata Properties Map & Building Analysis")
    
    # Create tabs for different map views
    map_tab1, map_tab2, map_tab3 = st.tabs(["🗺️ Interactive Map", "🔥 Density Heatmap", "⚖️ Scale Comparison"])
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    available_scales = sorted(df['strata_scale'].dropna().unique())
    selected_scales = st.sidebar.multiselect("Select Strata Scale(s):", options=available_scales, default=available_scales)
    
    min_price = int(df['saleslastsoldprice'].min())
    max_price = int(df['saleslastsoldprice'].max())
    price_range = st.sidebar.slider("Price Range ($):", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=10000)
    
    council_areas = sorted(df['councilarea'].dropna().unique())
    selected_councils = st.sidebar.multiselect("Council Area(s):", options=council_areas, default=["MARIBYRNONG"] if "MARIBYRNONG" in council_areas else council_areas[:1])
    
    # Apply filters
    df_filtered = df[
        (df['strata_scale'].isin(selected_scales)) &
        (df['saleslastsoldprice'] >= price_range[0]) &
        (df['saleslastsoldprice'] <= price_range[1]) &
        (df['councilarea'].isin(selected_councils))
    ]
    
    # Create spatial index for fast lookups
    spatial_index = create_spatial_index(df_filtered)
    
    # Initialize session state
    if 'selected_building_id' not in st.session_state:
        st.session_state.selected_building_id = None
    
    # Tab 1: Interactive Map
    with map_tab1:
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📍 Interactive Map")
            st.caption("Click on a building marker to view all strata units in that building")
            
            # Create and display map
            folium_map = create_folium_map(df_filtered, st.session_state.selected_building_id)
            map_data = st_folium(folium_map, width=700, height=500, returned_objects=["last_clicked"], key="strata_map")
            
            # Handle map clicks
            if map_data and map_data["last_clicked"]:
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]
                
                # Fast building lookup using spatial index
                clicked_building_id = find_building_from_coords(clicked_lat, clicked_lon, spatial_index)
                
                if clicked_building_id and clicked_building_id != st.session_state.selected_building_id:
                    st.session_state.selected_building_id = clicked_building_id
                    st.rerun()
        
        with col2:
            st.subheader("🏢 Building Information")
            
            if st.session_state.selected_building_id:
                # Get building data
                building_data = get_building_data(st.session_state.selected_building_id, df)
                
                if not building_data.empty:
                    # Building summary
                    summary = create_building_summary(building_data)
                    
                    st.success(f"Building Selected: {st.session_state.selected_building_id}")
                    
                    # Display summary as metrics
                    for key, value in summary.items():
                        st.metric(key, value)
                    
                    # Reset button
                    if st.button("🔄 Reset Selection", type="secondary", key="reset_tab1"):
                        st.session_state.selected_building_id = None
                        st.rerun()
                else:
                    st.warning("No data found for selected building")
            else:
                st.info("👆 Click on a building marker on the map to view its strata information")
                
                # Show general statistics
                if not df_filtered.empty:
                    st.subheader("📊 Current Filter Statistics")
                    st.metric("Total Properties", len(df_filtered))
                    st.metric("Unique Buildings", df_filtered['building_id'].nunique())
                    st.metric("Average Price", f"${df_filtered['saleslastsoldprice'].mean():,.0f}")

    # Tab 2: Density Heatmap
    with map_tab2:
        st.subheader("🔥 Strata Scale Density Heatmap")
        st.caption("Visualize the concentration and distribution of different strata scales")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### Heatmap Controls")
            
            # Scale selection for heatmap
            heatmap_scales = st.multiselect(
                "Select Scales to Display:",
                options=available_scales,
                default=available_scales,
                key="heatmap_scales"
            )
            
            # Heatmap statistics
            if not df_filtered.empty and heatmap_scales:
                st.markdown("#### Statistics")
                for scale in heatmap_scales:
                    scale_count = len(df_filtered[df_filtered['strata_scale'] == scale])
                    st.metric(f"{scale.title()} Properties", scale_count)
                
                # Density analysis
                st.markdown("#### Density Analysis")
                density_stats = df_filtered.groupby(['councilarea', 'strata_scale']).size().reset_index(name='count')
                
                for council in selected_councils:
                    council_stats = density_stats[density_stats['councilarea'] == council]
                    if not council_stats.empty:
                        st.markdown(f"**{council}:**")
                        for _, row in council_stats.iterrows():
                            st.write(f"• {row['strata_scale']}: {row['count']} properties")
        
        with col1:
            if heatmap_scales and not df_filtered.empty:
                # Create heatmap
                heatmap_fig = create_heatmap(df_filtered, heatmap_scales)
                st_folium(heatmap_fig, width=700, height=600, key="heatmap")
            else:
                st.info("Select at least one strata scale to display the heatmap")
                
        # Heatmap insights
        if not df_filtered.empty:
            st.markdown("---")
            st.subheader("📈 Heatmap Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Concentration by council area
                council_conc = df_filtered.groupby(['councilarea', 'strata_scale']).size().unstack(fill_value=0)
                fig_council = px.bar(
                    council_conc.reset_index(),
                    x='councilarea',
                    y=council_conc.columns.tolist(),
                    title="Scale Distribution by Council Area"
                )
                st.plotly_chart(fig_council, use_container_width=True)
            
            with col2:
                # Price distribution by scale
                fig_price = px.box(
                    df_filtered,
                    x='strata_scale',
                    y='saleslastsoldprice',
                    title="Price Distribution by Strata Scale"
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col3:
                # Geographic spread
                spread_stats = df_filtered.groupby('strata_scale').agg({
                    'addresslocation_lat': ['min', 'max'],
                    'addresslocation_lon': ['min', 'max'],
                    'building_id': 'nunique'
                }).round(4)
                
                st.markdown("**Geographic Spread:**")
                for scale in df_filtered['strata_scale'].unique():
                    buildings = df_filtered[df_filtered['strata_scale'] == scale]['building_id'].nunique()
                    st.write(f"**{scale.title()}:** {buildings} buildings")

    # Tab 3: Scale Comparison
    with map_tab3:
        st.subheader("⚖️ Small vs Medium Scale Comparison")
        
        if not df_filtered.empty:
            # Create comparison map
            comparison_map = create_comparison_heatmap(df_filtered)
            st_folium(comparison_map, width=1000, height=600, key="comparison_map")
            
            # Comparison statistics
            st.subheader("📊 Comparative Analysis")
            
            comparison_stats = df_filtered.groupby('strata_scale').agg({
                'building_id': 'nunique',
                'saleslastsoldprice': ['mean', 'median', 'std'],
                'addresssuburb': 'nunique',
                'councilarea': 'nunique'
            }).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Small Scale Properties")
                small_data = df_filtered[df_filtered['strata_scale'] == 'small scale']
                if not small_data.empty:
                    st.metric("Buildings", small_data['building_id'].nunique())
                    st.metric("Average Price", f"${small_data['saleslastsoldprice'].mean():,.0f}")
                    st.metric("Suburbs Covered", small_data['addresssuburb'].nunique())
                    
                    # Top suburbs for small scale
                    top_suburbs_small = small_data['addresssuburb'].value_counts().head(5)
                    st.markdown("**Top Suburbs:**")
                    for suburb, count in top_suburbs_small.items():
                        st.write(f"• {suburb}: {count} properties")
                else:
                    st.info("No small scale properties in current selection")
            
            with col2:
                st.markdown("#### Medium Scale Properties")
                medium_data = df_filtered[df_filtered['strata_scale'] == 'medium scale']
                if not medium_data.empty:
                    st.metric("Buildings", medium_data['building_id'].nunique())
                    st.metric("Average Price", f"${medium_data['saleslastsoldprice'].mean():,.0f}")
                    st.metric("Suburbs Covered", medium_data['addresssuburb'].nunique())
                    
                    # Top suburbs for medium scale
                    top_suburbs_medium = medium_data['addresssuburb'].value_counts().head(5)
                    st.markdown("**Top Suburbs:**")
                    for suburb, count in top_suburbs_medium.items():
                        st.write(f"• {suburb}: {count} properties")
                else:
                    st.info("No medium scale properties in current selection")
            
            # Market insights
            st.markdown("---")
            st.subheader("💡 Market Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                # Price comparison chart
                if len(df_filtered['strata_scale'].unique()) > 1:
                    fig_comparison = px.violin(
                        df_filtered,
                        x='strata_scale',
                        y='saleslastsoldprice',
                        title="Price Distribution Comparison",
                        color='strata_scale'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
            
            with insights_col2:
                # Geographic distribution
                scale_geo = df_filtered.groupby(['councilarea', 'strata_scale']).size().reset_index(name='count')
                fig_geo = px.sunburst(
                    scale_geo,
                    path=['councilarea', 'strata_scale'],
                    values='count',
                    title="Geographic Distribution by Scale"
                )
                st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("No data available for comparison with current filters")
    
    # Main layout - only for Tab 1, moved inside the tab
    
    # Building details table (full width) - only for Tab 1
    with map_tab1:
        if st.session_state.selected_building_id:
            building_data = get_building_data(st.session_state.selected_building_id, df)
            
            if not building_data.empty:
                st.subheader(f"📋 All Units in Building {st.session_state.selected_building_id}")
                
                # Prepare display columns
                display_columns = [
                    'unit_number', 'type', 'saleslastsoldprice', 'saleslastsolddate',
                    'addresscomplete', 'strata_scale', 'councilarea'
                ]
                
                # Filter columns that exist in the dataframe
                available_columns = [col for col in display_columns if col in building_data.columns]
                
                # Create a clean display dataframe
                display_df = building_data[available_columns].copy()
                
                # Format price column
                if 'saleslastsoldprice' in display_df.columns:
                    display_df['saleslastsoldprice'] = display_df['saleslastsoldprice'].apply(lambda x: f"${x:,.0f}")
                
                # Rename columns for better display
                column_rename = {
                    'unit_number': 'Unit #',
                    'type': 'Property Type',
                    'saleslastsoldprice': 'Last Sale Price',
                    'saleslastsolddate': 'Sale Date',
                    'addresscomplete': 'Address',
                    'strata_scale': 'Strata Scale',
                    'councilarea': 'Council Area'
                }
                
                display_df = display_df.rename(columns=column_rename)
                
                # Display the table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Additional analysis for the building
                if len(building_data) > 1:
                    st.subheader("📈 Building Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Price distribution
                        fig_hist = px.histogram(
                            building_data, 
                            x='saleslastsoldprice', 
                            title=f'Price Distribution - Building {st.session_state.selected_building_id}',
                            nbins=min(10, len(building_data))
                        )
                        fig_hist.update_layout(height=300)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Property type distribution
                        type_counts = building_data['type'].value_counts()
                        fig_pie = px.pie(
                            values=type_counts.values, 
                            names=type_counts.index,
                            title='Property Types'
                        )
                        fig_pie.update_layout(height=300)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col3:
                        # Price vs Unit scatter (if unit numbers are numeric)
                        try:
                            building_data_numeric = building_data.copy()
                            building_data_numeric['unit_number_numeric'] = pd.to_numeric(building_data_numeric['unit_number'], errors='coerce')
                            
                            if not building_data_numeric['unit_number_numeric'].isna().all():
                                fig_scatter = px.scatter(
                                    building_data_numeric,
                                    x='unit_number_numeric',
                                    y='saleslastsoldprice',
                                    title='Price vs Unit Number',
                                    hover_data=['type']
                                )
                                fig_scatter.update_layout(height=300)
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.info("Unit numbers are not numeric for detailed analysis")
                        except:
                            st.info("Could not create unit number analysis")
if menu == "Chatbot":
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hi, I'm a friendly assistant. How can I help you?")]
    if "context_log" not in st.session_state:
        st.session_state.context_log = ["Retrieved context will be displayed here"]

    # Define the agent
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api_key)
    agent = create_csv_agent(
        llm, 
        "./completed_data_stratxcel.csv", 
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        system_message="You are a real estate expertise in managing small and medium scale strata. Gives recommendation and answer based on the related data and using a statsmodel when you ask some relationship or any statistical method appropriate for user query"
    )

    def query_data(query):
        return {"output": f"This is a placeholder response to: {query}"}

    input_mode = st.selectbox("Choose your input method:", ("Text", "Speech"))
    
    # Context toggle
    result = st.toggle("Toggle Context")
    if result:
        st.write(st.session_state.context_log)

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    if input_mode == "Text":
        user_query = st.chat_input("Type your message here...")
        
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            # Show human message
            with st.chat_message("Human"):
                st.markdown(user_query)

            # Process response
            try:
                raw_response = query_data(user_query)
                if isinstance(raw_response, dict) and 'output' in raw_response:
                    response_text = raw_response['output']
                elif isinstance(raw_response, str):
                    response_text = raw_response
                else:
                    response_text = "Sorry, I couldn't generate a useful answer."

            except Exception as e:
                import traceback
                response_text = "An internal error occurred:\n" + traceback.format_exc()

            # Show AI response and add to history
            with st.chat_message("AI"):
                st.markdown(response_text)
            
            st.session_state.chat_history.append(AIMessage(content=response_text))

    else:  # Speech mode
        st.write("Click the button to record your speech:")
        audio_bytes = audio_recorder(key="speech_recorder")  # Add unique key
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            try:
                from io import BytesIO

                recognizer = sr.Recognizer()
                with sr.AudioFile(BytesIO(audio_bytes)) as source:
                    audio = recognizer.record(source)
                    user_query = recognizer.recognize_google(audio)
                    st.write("Recognized Text:", user_query)

                if user_query:
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    
                    with st.chat_message("Human"):
                        st.markdown(user_query)
                    
                    with st.spinner("Thinking...and Calculating"):
                        response = query_data(user_query)
                    
                    response_text = response['output'] if isinstance(response, dict) and 'output' in response else str(response)
                    
                    with st.chat_message("AI"):
                        st.markdown(response_text)
                    
                    st.session_state.chat_history.append(AIMessage(content=response_text))

            except Exception as e:
                st.write("Could not process audio:", str(e))

# Footer
st.markdown("---")
st.markdown("Built with love and passion ❤‍🔥 from Group 15")