import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from math import radians, sin, cos, sqrt, atan2
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
from geopy.distance import geodesic
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
        
        # Create links - MODIFIED to use building_id count
        source = []
        target = []
        value = []
        link_colors = []
        
        # Flow 1: strata_scale -> councilarea (count unique buildings)
        flow1 = filtered_df.groupby(['strata_scale', 'councilarea'])['building_id'].nunique().reset_index(name='count')
        for _, row in flow1.iterrows():
            source.append(node_dict[row['strata_scale']])
            target.append(node_dict[row['councilarea']])
            value.append(row['count'])
            link_colors.append('rgba(231, 76, 60, 0.4)')
        
        # Flow 2: councilarea -> addresssuburb (count unique buildings)
        flow2 = filtered_df.groupby(['councilarea', 'addresssuburb'])['building_id'].nunique().reset_index(name='count')
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
        
        # Update title to reflect building count
        unique_buildings = filtered_df['building_id'].nunique()
        fig.update_layout(
            title={
                'text': f"Building Flow Analysis ({unique_buildings} unique buildings)",
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
        """Create comprehensive summary tables - MODIFIED to use building_id"""
        if filtered_df.empty:
            return None, None, None, None
        
        try:
            # 1. Council Area Summary - count unique buildings
            council_summary = filtered_df.groupby('councilarea').agg({
                'building_id': 'nunique',  # Count unique buildings instead of rows
                'saleslastsoldprice': ['mean', 'median'],
                'type': lambda x: x.value_counts().to_dict(),
                'strata_scale': lambda x: x.value_counts().to_dict()
            }).round(0)
            
            council_summary.columns = ['Building_Count', 'Avg_Price', 'Median_Value', 'Property_Types', 'Strata_Scales']
            council_summary = council_summary.reset_index()
            
            # 2. Suburb Summary - count unique buildings
            suburb_summary = filtered_df.groupby(['councilarea', 'addresssuburb']).agg({
                'building_id': 'nunique',  # Count unique buildings
                'saleslastsoldprice': ['mean', 'median'],
                'type': lambda x: ', '.join(x.value_counts().index[:3]) if len(x.value_counts()) > 0 else 'N/A'
            }).round(0)
            
            suburb_summary.columns = ['Building_Count', 'Avg_Price', 'Median_Value', 'Main_Types']
            suburb_summary = suburb_summary.reset_index()
            
            # 3. Property Type Summary - count unique buildings
            type_summary = filtered_df.groupby(['councilarea', 'type']).agg({
                'building_id': 'nunique',  # Count unique buildings
                'saleslastsoldprice': 'mean',
                'strata_scale': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
            }).round(0)
            
            type_summary.columns = ['Building_Count', 'Avg_Price', 'Common_Scale']
            type_summary = type_summary.reset_index()
            
            # 4. Strata Scale Summary - count unique buildings
            strata_summary = filtered_df.groupby(['councilarea', 'strata_scale']).agg({
                'building_id': 'nunique',  # Count unique buildings
                'saleslastsoldprice': ['mean', 'median', 'max']
            }).round(0)
            
            strata_summary.columns = ['Building_Count', 'Avg_Price', 'Median_Price', 'Max_Price']
            strata_summary = strata_summary.reset_index()
            
            return council_summary, suburb_summary, type_summary, strata_summary
            
        except Exception as e:
            st.error(f"Error creating summary tables: {str(e)}")
            return None, None, None, None

    def analyze_top_suburbs(filtered_df, top_n=10):
        """Analyze top suburbs by strata scale with comprehensive metrics - MODIFIED to use building_id"""
        if filtered_df.empty:
            return None, None, None
        
        # Overall suburb analysis - count unique buildings
        suburb_analysis = filtered_df.groupby('addresssuburb').agg({
            'building_id': 'nunique',  # Total unique buildings
            'saleslastsoldprice': ['mean', 'median'],
            'strata_scale': [
                lambda x: filtered_df[(filtered_df['addresssuburb'] == x.name) & (filtered_df['strata_scale'] == 'small scale')]['building_id'].nunique(),
                lambda x: filtered_df[(filtered_df['addresssuburb'] == x.name) & (filtered_df['strata_scale'] == 'medium scale')]['building_id'].nunique(),
                lambda x: x.value_counts().to_dict()
            ],
            'councilarea': 'first'
        }).round(0)
        
        # Fix the small and medium scale counting
        small_scale_counts = []
        medium_scale_counts = []
        for suburb in suburb_analysis.index:
            small_count = filtered_df[(filtered_df['addresssuburb'] == suburb) & 
                                    (filtered_df['strata_scale'] == 'small scale')]['building_id'].nunique()
            medium_count = filtered_df[(filtered_df['addresssuburb'] == suburb) & 
                                     (filtered_df['strata_scale'] == 'medium scale')]['building_id'].nunique()
            small_scale_counts.append(small_count)
            medium_scale_counts.append(medium_count)
        
        suburb_analysis.columns = [
            'Total_Buildings', 'Avg_Price', 'Median_Price',
            'Small_Scale_Count_Old', 'Medium_Scale_Count_Old', 'Scale_Distribution',
            'Council_Area'
        ]
        
        # Replace with correct counts
        suburb_analysis['Small_Scale_Count'] = small_scale_counts
        suburb_analysis['Medium_Scale_Count'] = medium_scale_counts
        suburb_analysis = suburb_analysis.drop(['Small_Scale_Count_Old', 'Medium_Scale_Count_Old'], axis=1)
        suburb_analysis = suburb_analysis.reset_index()
        
        # Calculate dominance scores
        suburb_analysis['Small_Scale_Dominance'] = (
            suburb_analysis['Small_Scale_Count'] / suburb_analysis['Total_Buildings'] * 100
        ).round(1)
        suburb_analysis['Medium_Scale_Dominance'] = (
            suburb_analysis['Medium_Scale_Count'] / suburb_analysis['Total_Buildings'] * 100
        ).round(1)
        
        # Separate analysis for small and medium scale
        small_scale_suburbs = suburb_analysis[suburb_analysis['Small_Scale_Count'] > 0].copy()
        small_scale_suburbs = small_scale_suburbs.sort_values('Small_Scale_Count', ascending=False).head(top_n)
        
        medium_scale_suburbs = suburb_analysis[suburb_analysis['Medium_Scale_Count'] > 0].copy()
        medium_scale_suburbs = medium_scale_suburbs.sort_values('Medium_Scale_Count', ascending=False).head(top_n)
        
        # Overall top suburbs (by total buildings)
        top_suburbs_overall = suburb_analysis.sort_values('Total_Buildings', ascending=False).head(top_n)
        
        return small_scale_suburbs, medium_scale_suburbs, top_suburbs_overall

    def create_suburb_comparison_charts(filtered_df):
        """Create comprehensive suburb comparison charts - MODIFIED to use building_id"""
        if filtered_df.empty:
            return None, None, None, None
        
        # 1. Top suburbs by strata scale - count unique buildings
        suburb_scale_data = filtered_df.groupby(['addresssuburb', 'strata_scale'])['building_id'].nunique().reset_index(name='building_count')
        suburb_scale_pivot = suburb_scale_data.pivot(index='addresssuburb', columns='strata_scale', values='building_count').fillna(0)
        suburb_scale_pivot['total'] = suburb_scale_pivot.sum(axis=1)
        top_suburbs_for_chart = suburb_scale_pivot.nlargest(15, 'total')
        
        fig1 = px.bar(
            top_suburbs_for_chart.reset_index(),
            x='addresssuburb',
            y=['small scale', 'medium scale'] if 'small scale' in top_suburbs_for_chart.columns and 'medium scale' in top_suburbs_for_chart.columns else top_suburbs_for_chart.columns[:-1],
            title="Top 15 Suburbs by Strata Scale Distribution (Unique Buildings)",
            labels={'value': 'Number of Buildings', 'addresssuburb': 'Suburb'},
            color_discrete_map={'small scale': '#FF6B6B', 'medium scale': '#4ECDC4'}
        )
        fig1.update_layout(xaxis_tickangle=-45, height=500)
        
        # 2. Price comparison by suburb and scale
        price_data = filtered_df.groupby(['addresssuburb', 'strata_scale'])['saleslastsoldprice'].mean().reset_index()
        # Get top suburbs by building count for price comparison
        top_building_suburbs = filtered_df.groupby('addresssuburb')['building_id'].nunique().nlargest(10).index
        price_data_filtered = price_data[price_data['addresssuburb'].isin(top_building_suburbs)]
        
        fig2 = px.bar(
            price_data_filtered,
            x='addresssuburb',
            y='saleslastsoldprice',
            color='strata_scale',
            title="Average Price by Suburb and Strata Scale (Top 10 by Building Count)",
            labels={'saleslastsoldprice': 'Average Sale Price ($)', 'addresssuburb': 'Suburb'},
            color_discrete_map={'small scale': '#FF6B6B', 'medium scale': '#4ECDC4'}
        )
        fig2.update_layout(xaxis_tickangle=-45, height=500)
        
        # 3. Suburb concentration heatmap - count unique buildings
        suburb_council_data = filtered_df.groupby(['councilarea', 'addresssuburb', 'strata_scale'])['building_id'].nunique().reset_index(name='building_count')
        fig3 = px.treemap(
            suburb_council_data,
            path=['councilarea', 'addresssuburb', 'strata_scale'],
            values='building_count',
            title="Building Distribution Hierarchy: Council → Suburb → Scale",
            color='strata_scale',
            color_discrete_map={'small scale': '#FF6B6B', 'medium scale': '#4ECDC4'}
        )
        fig3.update_layout(height=600)
        
        # 4. Market share analysis - based on building count
        total_buildings = filtered_df['building_id'].nunique()
        market_share = filtered_df.groupby('addresssuburb')['building_id'].nunique().reset_index(name='building_count')
        market_share['market_share'] = (market_share['building_count'] / total_buildings * 100).round(2)
        top_market_share = market_share.nlargest(10, 'market_share')
        
        fig4 = px.pie(
            top_market_share,
            values='market_share',
            names='addresssuburb',
            title="Top 10 Suburbs Market Share by Buildings (%)"
        )
        fig4.update_layout(height=500)
        
        return fig1, fig2, fig3, fig4

    st.title("🏢 Strata Building Analysis Dashboard")
    st.markdown("### Council Area & Suburb Analysis")
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Get unique values for filters
    unique_councils = sorted(df['councilarea'].unique())
    unique_suburbs = sorted(df['addresssuburb'].unique())
    # Strata scale filter
    selected_scales = st.sidebar.multiselect(
        "Strata Scales",
        options=sorted(df['strata_scale'].unique()),
        default=sorted(df['strata_scale'].unique())
    )
    
    # Council area filter
    st.sidebar.markdown("**Select Council Areas:**")
    select_all_councils = st.sidebar.checkbox("Select All Council Areas", value=True)
    
    if select_all_councils:
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
    
    # Suburb filter - MODIFIED to use building_id for top calculations
    st.sidebar.markdown("**Select Suburbs:**")
    suburb_filter_options = ["All Suburbs", "Top Small Scale", "Top Medium Scale", "Custom Selection"]
    suburb_filter_mode = st.sidebar.selectbox("Suburb Filter Mode:", suburb_filter_options)
    
    if suburb_filter_mode == "All Suburbs":
        # Filter suburbs based on selected councils
        available_suburbs = sorted(df[df['councilarea'].isin(selected_councils)]['addresssuburb'].unique()) if selected_councils else unique_suburbs
        selected_suburbs = st.sidebar.multiselect(
            "Suburbs (All selected)",
            options=available_suburbs,
            default=available_suburbs,
            disabled=True
        )
    elif suburb_filter_mode == "Top Small Scale":
        top_n_small = st.sidebar.slider("Number of top small scale suburbs:", 5, 20, 10)
        # Get top suburbs for small scale based on unique building count
        temp_df = df[df['councilarea'].isin(selected_councils)] if selected_councils else df
        small_scale_df = temp_df[temp_df['strata_scale'] == 'small scale']
        top_small_suburbs = small_scale_df.groupby('addresssuburb')['building_id'].nunique().nlargest(top_n_small).index.tolist()
        selected_suburbs = st.sidebar.multiselect(
            f"Top {top_n_small} Small Scale Suburbs (by buildings)",
            options=top_small_suburbs,
            default=top_small_suburbs,
            disabled=True
        )
    elif suburb_filter_mode == "Top Medium Scale":
        top_n_medium = st.sidebar.slider("Number of top medium scale suburbs:", 5, 20, 10)
        # Get top suburbs for medium scale based on unique building count
        temp_df = df[df['councilarea'].isin(selected_councils)] if selected_councils else df
        medium_scale_df = temp_df[temp_df['strata_scale'] == 'medium scale']
        top_medium_suburbs = medium_scale_df.groupby('addresssuburb')['building_id'].nunique().nlargest(top_n_medium).index.tolist()
        selected_suburbs = st.sidebar.multiselect(
            f"Top {top_n_medium} Medium Scale Suburbs (by buildings)",
            options=top_medium_suburbs,
            default=top_medium_suburbs,
            disabled=True
        )
    else:  # Custom Selection
        available_suburbs = sorted(df[df['councilarea'].isin(selected_councils)]['addresssuburb'].unique()) if selected_councils else unique_suburbs
        selected_suburbs = st.sidebar.multiselect(
            "Select Suburbs:",
            options=available_suburbs,
            default=available_suburbs[:5] if len(available_suburbs) >= 5 else available_suburbs
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
    if selected_councils and selected_suburbs:
        filtered_df = df[
            (df['councilarea'].isin(selected_councils)) &
            (df['addresssuburb'].isin(selected_suburbs)) &
            (df['saleslastsoldprice'] >= min_price) &
            (df['saleslastsoldprice'] <= max_price) &
            (df['type'].isin(selected_types)) &
            (df['strata_scale'].isin(selected_scales))
        ]
    else:
        st.warning("⚠️ Please select at least one council area and suburb.")
        filtered_df = pd.DataFrame()
    
    if not filtered_df.empty:
        # Display key metrics - MODIFIED to show building count
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_properties = len(filtered_df)
            st.metric("Total Properties", total_properties)
        
        with col2:
            avg_price = filtered_df['saleslastsoldprice'].mean()
            st.metric("Average Price", f"${avg_price:,.0f}")
        
        with col3:
            median_value = filtered_df['saleslastsoldprice'].median()
            st.metric("Median Price", f"${median_value:,.0f}")
        
        with col4:
            unique_suburbs_count = filtered_df['addresssuburb'].nunique()
            st.metric("Unique Suburbs", unique_suburbs_count)
        
        with col5:
            unique_buildings = filtered_df['building_id'].nunique()
            st.metric("Unique Buildings", unique_buildings)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Tables", "🏘️ Suburb Analysis", "🌊 Sankey Diagram", "📈 Charts"])
        
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
                    st.dataframe(display_council[['councilarea', 'Building_Count', 'Avg_Price', 'Median_Value']], 
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
            st.markdown("## 🏘️ Top Suburb Analysis")
            
            # Analyze top suburbs
            small_scale_suburbs, medium_scale_suburbs, top_suburbs_overall = analyze_top_suburbs(filtered_df)
            
            if small_scale_suburbs is not None:
                # Display top suburbs analysis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🏠 Top Small Scale Suburbs")
                    if not small_scale_suburbs.empty:
                        display_small = small_scale_suburbs[['addresssuburb', 'Small_Scale_Count', 'Small_Scale_Dominance', 'Avg_Price', 'Council_Area']].copy()
                        display_small['Avg_Price'] = display_small['Avg_Price'].apply(lambda x: f"${x:,.0f}")
                        display_small.columns = ['Suburb', 'Small Scale Buildings', 'Dominance %', 'Avg Price', 'Council']
                        st.dataframe(display_small, hide_index=True, use_container_width=True)
                    else:
                        st.info("No small scale properties found")
                
                with col2:
                    st.markdown("### 🏢 Top Medium Scale Suburbs")
                    if not medium_scale_suburbs.empty:
                        display_medium = medium_scale_suburbs[['addresssuburb', 'Medium_Scale_Count', 'Medium_Scale_Dominance', 'Avg_Price', 'Council_Area']].copy()
                        display_medium['Avg_Price'] = display_medium['Avg_Price'].apply(lambda x: f"${x:,.0f}")
                        display_medium.columns = ['Suburb', 'Medium Scale Buildings', 'Dominance %', 'Avg Price', 'Council']
                        st.dataframe(display_medium, hide_index=True, use_container_width=True)
                    else:
                        st.info("No medium scale properties found")
                
                with col3:
                    st.markdown("### 🌟 Overall Top Suburbs")
                    if not top_suburbs_overall.empty:
                        display_overall = top_suburbs_overall[['addresssuburb', 'Total_Buildings', 'Small_Scale_Count', 'Medium_Scale_Count', 'Avg_Price']].copy()
                        display_overall['Avg_Price'] = display_overall['Avg_Price'].apply(lambda x: f"${x:,.0f}")
                        display_overall.columns = ['Suburb', 'Total Buildings', 'Small Scale', 'Medium Scale', 'Avg Price']
                        st.dataframe(display_overall, hide_index=True, use_container_width=True)
                
                # Suburb comparison charts
                st.markdown("---")
                st.markdown("### 📊 Suburb Comparison Charts")
                
                fig1, fig2, fig3, fig4 = create_suburb_comparison_charts(filtered_df)
                
                if fig1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.plotly_chart(fig4, use_container_width=True)
                    with col4:
                        # Key insights - MODIFIED for building counts
                        st.markdown("#### 🔍 Key Insights")
                        
                        if not small_scale_suburbs.empty:
                            top_small_suburb = small_scale_suburbs.iloc[0]
                            st.success(f"**Top Small Scale Suburb:** {top_small_suburb['addresssuburb']} with {int(top_small_suburb['Small_Scale_Count'])} buildings ({top_small_suburb['Small_Scale_Dominance']}% dominance)")
                        
                        if not medium_scale_suburbs.empty:
                            top_medium_suburb = medium_scale_suburbs.iloc[0]
                            st.info(f"**Top Medium Scale Suburb:** {top_medium_suburb['addresssuburb']} with {int(top_medium_suburb['Medium_Scale_Count'])} buildings ({top_medium_suburb['Medium_Scale_Dominance']}% dominance)")
                        
                        if not top_suburbs_overall.empty:
                            overall_top = top_suburbs_overall.iloc[0]
                            st.warning(f"**Most Active Suburb:** {overall_top['addresssuburb']} with {int(overall_top['Total_Buildings'])} total buildings")
                        
                        # Market concentration - based on buildings
                        total_suburbs = filtered_df['addresssuburb'].nunique()
                        top_5_concentration = (top_suburbs_overall.head(5)['Total_Buildings'].sum() / filtered_df['building_id'].nunique() * 100).round(1) if not top_suburbs_overall.empty else 0
                        st.metric("Top 5 Suburbs Building Share", f"{top_5_concentration}%")
                    
                    # Full width treemap
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No data available for suburb analysis with current filters.")
        
        with tab3:
            st.markdown("## 🌊 Sankey Diagram")
            sankey_fig = create_sankey_diagram(filtered_df)
            if sankey_fig:
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")
        
        with tab4:
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
                type_counts = filtered_df.groupby('type')['building_id'].nunique()
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
        #Helper Function
        def get_council_centers():
            """
            Returns approximate coordinates of council administrative centers
            You'll need to research and add the actual coordinates for each council area
            """
            council_centers = {
                'MARIBYRNONG': (-37.7745, 144.8932),
                'MELBOURNE': (-37.8136, 144.9631),
                'YARRA': (-37.8079, 144.9841),
                'PORT PHILLIP': (-37.8434, 144.9624),
                'STONNINGTON': (-37.8578, 145.0135),
                'GLEN EIRA': (-37.8893, 145.0368),
                'BAYSIDE': (-37.9134, 145.0187),
                'KINGSTON': (-38.0167, 145.1167),
                'MONASH': (-37.8864, 145.1361),
                'WHITEHORSE': (-37.8167, 145.1833),
                'MANNINGHAM': (-37.7833, 145.2167),
                'DAREBIN': (-37.7500, 145.0167),
                'BANYULE': (-37.7167, 145.0833),
                'NILLUMBIK': (-37.6333, 145.2167),
                'WHITTLESEA': (-37.5167, 145.1167),
                'MITCHELL': (-37.4000, 145.0000),
                'MURRINDINDI': (-37.3000, 145.7500),
                'MACEDON RANGES': (-37.4167, 144.5833),
                'HUME': (-37.6500, 144.8833),
                'MORELAND': (-37.7333, 144.9500),
                'MOONEE VALLEY': (-37.7500, 144.9167),
                'HOBSONS BAY': (-37.8500, 144.8333),
                'WYNDHAM': (-37.8500, 144.6667),
                'MELTON': (-37.6833, 144.5833),
                'BRIMBANK': (-37.7500, 144.7500),
                'BOROONDARA': (-37.8167, 145.0500),
                'KNOX': (-37.8667, 145.2500),
                'MAROONDAH': (-37.8167, 145.2500),
                'YARRA RANGES': (-37.7500, 145.5833),
                'CARDINIA': (-38.0833, 145.4500),
                'CASEY': (-38.0833, 145.3333),
                'GREATER DANDENONG': (-37.9833, 145.2167),
                'FRANKSTON': (-38.1333, 145.1167),
                'MORNINGTON PENINSULA': (-38.3500, 144.9500)
            }
            return council_centers

        def calculate_distances_to_centers(df, council_centers):
            """
            Calculate distance from each property to nearest council center
            """
            
            df_copy = df.copy()
            distances = []
            nearest_councils = []
            
            for idx, row in df_copy.iterrows():
                property_coords = (row['addresslocation_lat'], row['addresslocation_lon'])
                min_distance = float('inf')
                nearest_council = None
                
                for council, center_coords in council_centers.items():
                    distance = geodesic(property_coords, center_coords).kilometers
                    if distance < min_distance:
                        min_distance = distance
                        nearest_council = council
                
                distances.append(min_distance)
                nearest_councils.append(nearest_council)
            
            df_copy['distance_to_center'] = distances
            df_copy['nearest_council'] = nearest_councils
            
            return df_copy

        def calculate_dispersion_index(df):
            """
            Calculate spatial dispersion index for each council and strata scale
            """
            dispersion_data = []
            
            for council in df['councilarea'].unique():
                council_data = df[df['councilarea'] == council]
                
                for scale in council_data['strata_scale'].unique():
                    scale_data = council_data[council_data['strata_scale'] == scale]
                    
                    if len(scale_data) > 1:
                        # Calculate standard deviation of coordinates as dispersion measure
                        # lat_std = scale_data['addresslocation_lat'].std()
                        # lon_std = scale_data['addresslocation_lon'].std()
                        # dispersion_index = (lat_std + lon_std) * 100  # Scale for visibility
                        def haversine_distance(lat1, lon1, lat2, lon2):
                            R = 6371.0  # Earth radius in km
                            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

                            dlat = lat2 - lat1
                            dlon = lon2 - lon1

                            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
                            c = 2 * atan2(sqrt(a), sqrt(1 - a))

                            return R * c

                        def average_pairwise_haversine(latitudes, longitudes):
                            n = len(latitudes)
                            if n < 2:
                                return 0

                            distances = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    dist = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                                    distances.append(dist)

                            return np.mean(distances)
                        dispersion_index = average_pairwise_haversine(scale_data['addresslocation_lat'].tolist(), scale_data['addresslocation_lon'].tolist())
                    else:
                        dispersion_index = 0
                    
                    dispersion_data.append({
                        'councilarea': council,
                        'strata_scale': scale,
                        'dispersion_index': dispersion_index
                    })
            
            return pd.DataFrame(dispersion_data)

        def create_enhanced_heatmap(df, scales, councils, radius=None, show_rings=False):
            """
            Create enhanced heatmap with council centers and distance rings
            """
            # Start with your existing create_heatmap function
            m = create_heatmap(df, scales)  # Your existing function
            
            # Add council centers
            council_centers = get_council_centers()
            
            for council in councils:
                if council in council_centers:
                    center_coords = council_centers[council]
                    folium.Marker(
                        center_coords,
                        popup=f"{council} Council Center",
                        icon=folium.Icon(color='red', icon='building', prefix='fa'),
                        tooltip=f"{council} Administrative Center"
                    ).add_to(m)
                    
                    if show_rings and radius:
                        # Add distance rings
                        for r in [radius * 0.5, radius, radius * 1.5]:
                            folium.Circle(
                                center_coords,
                                radius=r * 1000,  # Convert km to meters
                                color='red',
                                weight=1,
                                opacity=0.3,
                                fill=False,
                                popup=f"{r}km radius"
                            ).add_to(m)
            
            return m

        def analyze_council_clustering(df, radius):
            """
            Analyze property clustering around council centers
            """
            council_centers = get_council_centers()
            clustering_data = []
            
            within_radius = 0
            outside_radius = 0
            
            for idx, row in df.iterrows():
                property_coords = (row['addresslocation_lat'], row['addresslocation_lon'])
                
                within_any_center = False
                for center_coords in council_centers.values():
                    distance = geodesic(property_coords, center_coords).kilometers
                    if distance <= radius:
                        within_any_center = True
                        break
                
                if within_any_center:
                    within_radius += 1
                else:
                    outside_radius += 1
            
            return pd.DataFrame([
                {'cluster_type': f'Within {radius}km of Council Centers', 'property_count': within_radius},
                {'cluster_type': f'Beyond {radius}km of Council Centers', 'property_count': outside_radius}
            ])

        def calculate_council_efficiency(df):
            """
            Calculate efficiency metrics for each council area
            """
            efficiency_data = []
            
            for council in df['councilarea'].unique():
                council_data = df[df['councilarea'] == council]
                
                # Calculate various efficiency metrics
                total_properties = len(council_data)
                avg_price = council_data['saleslastsoldprice'].mean()
                scale_diversity = council_data['strata_scale'].nunique()
                
                efficiency_data.append({
                    'Council Area': council,
                    'Total Properties': total_properties,
                    'Avg Price ($)': f"${avg_price:,.0f}",
                    'Scale Diversity': scale_diversity,
                    'Efficiency Score': (scale_diversity * total_properties) / (avg_price / 1000000)  # Custom metric
                })
            
            return pd.DataFrame(efficiency_data)
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
            
            # NEW: Council center analysis toggle
            show_council_centers = st.checkbox(
                "Show Council Centers",
                value=True,
                help="Display council administrative centers and distance analysis"
            )
            
            # NEW: Distance analysis options
            if show_council_centers:
                st.markdown("#### Distance Analysis")
                distance_radius = st.slider(
                    "Analysis Radius (km):",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Radius around council centers for density analysis"
                )
                
                show_distance_rings = st.checkbox(
                    "Show Distance Rings",
                    value=False,
                    help="Display concentric circles around council centers"
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
                # Create enhanced heatmap with council centers
                if show_council_centers:
                    heatmap_fig = create_enhanced_heatmap(
                        df_filtered, 
                        heatmap_scales, 
                        selected_councils,
                        distance_radius if show_council_centers else None,
                        show_distance_rings if show_council_centers else False
                    )
                else:
                    heatmap_fig = create_heatmap(df_filtered, heatmap_scales)
                
                st_folium(heatmap_fig, width=700, height=600, key="heatmap")
            else:
                st.info("Select at least one strata scale to display the heatmap")
        
        # NEW: Council Center Distance Analysis
        if show_council_centers and not df_filtered.empty:
            st.markdown("---")
            st.subheader("🎯 Council Center Distance Analysis")
            
            # Calculate distances to council centers
            council_centers = get_council_centers()  # You'll need to implement this
            df_with_distances = calculate_distances_to_centers(df_filtered, council_centers)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distance distribution
                fig_dist = px.histogram(
                    df_with_distances,
                    x='distance_to_center',
                    color='strata_scale',
                    title="Distance Distribution from Council Centers",
                    labels={'distance_to_center': 'Distance to Nearest Council Center (km)'},
                    marginal="box"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Council center proximity analysis
                proximity_stats = df_with_distances.groupby(['nearest_council', 'strata_scale']).agg({
                    'distance_to_center': ['mean', 'min', 'max'],
                    'saleslastsoldprice': 'mean',
                    'building_id': 'count'
                }).round(2)
                
                st.markdown("**Proximity Statistics:**")
                st.dataframe(proximity_stats, use_container_width=True)
        
        # Enhanced Heatmap insights
        if not df_filtered.empty:
            st.markdown("---")
            st.subheader("📈 Enhanced Heatmap Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Concentration by council area
                council_conc = df_filtered.groupby(['councilarea', 'strata_scale']).size().unstack(fill_value=0)
                fig_council = px.bar(
                    council_conc.reset_index(),
                    x='councilarea',
                    y=council_conc.columns.tolist(),
                    title="Scale Distribution by Council Area"
                )
                fig_council.update_xaxes(tickangle=45)
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
                # NEW: Dispersion index by council
                dispersion_data = calculate_dispersion_index(df_filtered)
                fig_dispersion = px.bar(
                    dispersion_data,
                    x='councilarea',
                    y='dispersion_index',
                    color='strata_scale',
                    title="Spatial Dispersion Index",
                    labels={'dispersion_index': 'Dispersion Index (higher = more spread)'}
                )
                fig_dispersion.update_xaxes(tickangle=45)
                st.plotly_chart(fig_dispersion, use_container_width=True)
                
            with col4:
                # Geographic spread with NEW council center distances
                if show_council_centers and 'distance_to_center' in df_filtered.columns:
                    fig_center_dist = px.scatter(
                        df_filtered,
                        x='distance_to_center',
                        y='saleslastsoldprice',
                        color='strata_scale',
                        size='buildingsize',
                        title="Price vs Distance from Council Centers",
                        labels={'distance_to_center': 'Distance to Council Center (km)'}
                    )
                    st.plotly_chart(fig_center_dist, use_container_width=True)
                else:
                    # Original geographic spread
                    spread_stats = df_filtered.groupby('strata_scale').agg({
                        'addresslocation_lat': ['min', 'max'],
                        'addresslocation_lon': ['min', 'max'],
                        'building_id': 'nunique'
                    }).round(4)
                    
                    st.markdown("**Geographic Spread:**")
                    for scale in df_filtered['strata_scale'].unique():
                        buildings = df_filtered[df_filtered['strata_scale'] == scale]['building_id'].nunique()
                        st.write(f"**{scale.title()}:** {buildings} buildings")

        # NEW: Council Center Clustering Analysis
        if show_council_centers and not df_filtered.empty:
            st.markdown("---")
            st.subheader("🏛️ Council Center Clustering Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster analysis around council centers
                clustering_data = analyze_council_clustering(df_filtered, distance_radius)
                
                fig_cluster = px.pie(
                    clustering_data,
                    values='property_count',
                    names='cluster_type',
                    title=f"Property Distribution within {distance_radius}km of Council Centers"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            with col2:
                # Council area efficiency metrics
                efficiency_metrics = calculate_council_efficiency(df_filtered)
                
                st.markdown("**Council Area Efficiency Metrics:**")
                st.dataframe(efficiency_metrics, use_container_width=True)

    # Helper functions to implement:


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
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        system_message="You are a real estate expertise in managing small and medium scale strata. Gives recommendation and answer based on the related data and using a statsmodel when you ask some relationship or any statistical method appropriate for user query. Used a groupby building_id when you ask about the most medium or small scale strata question"
    )

    def query_data(query):
        try:
            result = agent.run(query)
            return {"output": result}
        except Exception as e:
            import traceback
            return {"output": "Error occurred:\n" + traceback.format_exc()}

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