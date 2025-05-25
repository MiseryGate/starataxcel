import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Sample data - expand this with more diverse data for better demonstration


df = pd.read_csv('./scaled_lot.csv')

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

def main():
    st.set_page_config(
        page_title="Property Analysis Dashboard",
        page_icon="🏢",
        layout="wide"
    )
    
    st.title("🏢 Property Analysis Dashboard")
    st.markdown("### Interactive Council Area Analysis")
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("Built with love and passion ❤‍🔥 from Group 15")

if __name__ == "__main__":
    main()