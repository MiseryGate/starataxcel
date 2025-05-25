import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

# Set page config
st.set_page_config(page_title="Strata Scale Filter Map", layout="wide")

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


def main():
    st.title("🏢 Strata Properties Map Filter")
    
    df = pd.read_csv('completed_data_stratxcel.csv')
    
    st.sidebar.header("🔍 Filters")
    available_scales = sorted(df['strata_scale'].dropna().unique())
    selected_scales = st.sidebar.multiselect("Select Strata Scale(s):", options=available_scales, default=available_scales)
    
    min_price = int(df['saleslastsoldprice'].min())
    max_price = int(df['saleslastsoldprice'].max())
    price_range = st.sidebar.slider("Price Range ($):", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=10000)
    
    council_areas = sorted(df['councilarea'].dropna().unique())
    selected_councils = st.sidebar.multiselect("Council Area(s):", options=council_areas, default=council_areas)
    
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

if __name__ == "__main__":
    main()