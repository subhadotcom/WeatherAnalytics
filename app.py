import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import os

from data_processor import WeatherDataProcessor
from weather_analyzer import WeatherAnalyzer
from visualizations import WeatherVisualizer
from utils import format_file_size, validate_date_range

# Page configuration
st.set_page_config(
    page_title="Weather Data Analysis - Hot & Cold Days",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    st.title("🌡️ Weather Data Analysis for Hot & Cold Days")
    st.markdown("Analyze weather sensor data to identify and visualize temperature patterns")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Temperature thresholds
        st.subheader("Temperature Thresholds")
        hot_threshold = st.number_input(
            "Hot Day Threshold (°C)", 
            value=30.0, 
            min_value=-50.0, 
            max_value=60.0,
            step=0.5,
            help="Temperature above which a day is considered hot"
        )
        
        cold_threshold = st.number_input(
            "Cold Day Threshold (°C)", 
            value=5.0, 
            min_value=-50.0, 
            max_value=60.0,
            step=0.5,
            help="Temperature below which a day is considered cold"
        )
        
        # Data filtering options
        st.subheader("Data Filters")
        enable_date_filter = st.checkbox("Enable Date Range Filter")
        
        if enable_date_filter:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
            if not validate_date_range(start_date, end_date):
                st.error("End date must be after start date")
                return
        else:
            start_date = None
            end_date = None
        
        # Temperature range filter
        enable_temp_filter = st.checkbox("Enable Temperature Range Filter")
        if enable_temp_filter:
            temp_range = st.slider(
                "Temperature Range (°C)",
                min_value=-50.0,
                max_value=60.0,
                value=(-20.0, 50.0),
                step=0.5
            )
        else:
            temp_range = None
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "📊 Analysis", "📈 Visualizations", "📋 Summary"])
    
    with tab1:
        st.header("Data Upload and Processing")
        
        # File upload options
        upload_method = st.radio(
            "Choose data input method:",
            ["Upload Files", "Use Sample Data"],
            horizontal=True
        )
        
        uploaded_files = None
        sample_data_selected = None
        
        if upload_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload weather log files",
                type=['csv', 'json', 'txt', 'log'],
                accept_multiple_files=True,
                help="Supports CSV, JSON, and text log formats"
            )
        else:
            # Sample data selection
            sample_files = {
                "Weather Stations CSV (40 records)": "sample_data/weather_data.csv",
                "Multi-Location JSON (16 records)": "sample_data/weather_data.json", 
                "Sensor Logs (50 records)": "sample_data/weather_sensors.log",
                "Mixed Format Text (13 records)": "sample_data/mixed_format.txt"
            }
            
            st.info("Using built-in sample data to avoid upload issues. This demonstrates the full functionality with real weather sensor data.")
            
            selected_samples = st.multiselect(
                "Select sample data files to analyze:",
                options=list(sample_files.keys()),
                default=["Weather Stations CSV (40 records)", "Sensor Logs (50 records)"],
                help="Choose one or more sample datasets - combining multiple files shows data merging capabilities"
            )
            
            if selected_samples:
                sample_data_selected = [sample_files[name] for name in selected_samples]
        
        if uploaded_files or sample_data_selected:
            st.subheader("Selected Files")
            
            # Display file information
            total_size = 0
            file_info = []
            
            if uploaded_files:
                for file in uploaded_files:
                    size = len(file.getvalue())
                    total_size += size
                    file_info.append({
                        'Name': file.name,
                        'Size': format_file_size(size),
                        'Type': file.name.split('.')[-1].upper()
                    })
            elif sample_data_selected:
                for file_path in sample_data_selected:
                    try:
                        size = os.path.getsize(file_path)
                        total_size += size
                        file_info.append({
                            'Name': os.path.basename(file_path),
                            'Size': format_file_size(size),
                            'Type': file_path.split('.')[-1].upper()
                        })
                    except:
                        file_info.append({
                            'Name': os.path.basename(file_path),
                            'Size': 'Unknown',
                            'Type': file_path.split('.')[-1].upper()
                        })
            
            df_files = pd.DataFrame(file_info)
            st.dataframe(df_files, use_container_width=True)
            if total_size > 0:
                st.info(f"Total size: {format_file_size(total_size)}")
            
            # Process data button
            if st.button("Process Data", type="primary"):
                if not uploaded_files and not sample_data_selected:
                    st.error("Please select files to process first.")
                    st.stop()
                with st.spinner("Processing weather data..."):
                    try:
                        processor = WeatherDataProcessor()
                        
                        # Process all files (uploaded or sample)
                        all_data = []
                        files_to_process = uploaded_files if uploaded_files else sample_data_selected
                        progress_bar = st.progress(0)
                        
                        for i, file in enumerate(files_to_process):
                            if uploaded_files:
                                # Handle uploaded files
                                file_content = file.getvalue()
                                filename = file.name
                            else:
                                # Handle sample files
                                try:
                                    with open(file, 'rb') as f:
                                        file_content = f.read()
                                    filename = os.path.basename(file)
                                except Exception as e:
                                    st.error(f"Error reading sample file {file}: {str(e)}")
                                    continue
                            
                            if filename.endswith('.csv'):
                                data = processor.process_csv(file_content)
                            elif filename.endswith('.json'):
                                data = processor.process_json(file_content)
                            else:
                                data = processor.process_text_log(file_content)
                            
                            all_data.append(data)
                            progress_bar.progress((i + 1) / len(files_to_process))
                        
                        # Combine all data
                        combined_data = pd.concat(all_data, ignore_index=True)
                        
                        # Apply filters
                        if enable_date_filter:
                            combined_data = processor.filter_by_date_range(
                                combined_data, start_date, end_date
                            )
                        
                        if enable_temp_filter:
                            combined_data = processor.filter_by_temperature_range(
                                combined_data, temp_range[0], temp_range[1]
                            )
                        
                        st.session_state.processed_data = combined_data
                        st.success(f"Successfully processed {len(combined_data)} weather records!")
                        
                        # Display sample data
                        st.subheader("Sample Data Preview")
                        st.dataframe(combined_data.head(10), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        
        # Data quality information
        if st.session_state.processed_data is not None:
            st.subheader("Data Quality Summary")
            data = st.session_state.processed_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(data))
            
            with col2:
                missing_temp = data['temperature'].isna().sum()
                st.metric("Missing Temperature", missing_temp)
            
            with col3:
                valid_dates = data['timestamp'].notna().sum()
                st.metric("Valid Timestamps", valid_dates)
            
            with col4:
                unique_locations = data['location'].nunique() if 'location' in data.columns else 0
                st.metric("Unique Locations", unique_locations)
    
    with tab2:
        st.header("Weather Analysis")
        
        if st.session_state.processed_data is None:
            st.warning("Please upload and process data first in the Data Upload tab.")
            return
        
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing weather patterns..."):
                try:
                    analyzer = WeatherAnalyzer(
                        hot_threshold=hot_threshold,
                        cold_threshold=cold_threshold
                    )
                    
                    results = analyzer.analyze(st.session_state.processed_data)
                    st.session_state.analysis_results = results
                    
                    st.success("Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        # Display analysis results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            st.subheader("Temperature Classification")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Hot Days", 
                    results['hot_days_count'],
                    delta=f"{results['hot_days_percentage']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Cold Days", 
                    results['cold_days_count'],
                    delta=f"{results['cold_days_percentage']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Normal Days", 
                    results['normal_days_count'],
                    delta=f"{results['normal_days_percentage']:.1f}%"
                )
            
            # Temperature statistics
            st.subheader("Temperature Statistics")
            
            stats_data = {
                'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                'All Data': [
                    f"{results['temp_stats']['mean']:.2f}°C",
                    f"{results['temp_stats']['median']:.2f}°C",
                    f"{results['temp_stats']['min']:.2f}°C",
                    f"{results['temp_stats']['max']:.2f}°C",
                    f"{results['temp_stats']['std']:.2f}°C"
                ],
                'Hot Days': [
                    f"{results['hot_stats']['mean']:.2f}°C" if results['hot_stats']['mean'] is not None else "N/A",
                    f"{results['hot_stats']['median']:.2f}°C" if results['hot_stats']['median'] is not None else "N/A",
                    f"{results['hot_stats']['min']:.2f}°C" if results['hot_stats']['min'] is not None else "N/A",
                    f"{results['hot_stats']['max']:.2f}°C" if results['hot_stats']['max'] is not None else "N/A",
                    f"{results['hot_stats']['std']:.2f}°C" if results['hot_stats']['std'] is not None else "N/A"
                ],
                'Cold Days': [
                    f"{results['cold_stats']['mean']:.2f}°C" if results['cold_stats']['mean'] is not None else "N/A",
                    f"{results['cold_stats']['median']:.2f}°C" if results['cold_stats']['median'] is not None else "N/A",
                    f"{results['cold_stats']['min']:.2f}°C" if results['cold_stats']['min'] is not None else "N/A",
                    f"{results['cold_stats']['max']:.2f}°C" if results['cold_stats']['max'] is not None else "N/A",
                    f"{results['cold_stats']['std']:.2f}°C" if results['cold_stats']['std'] is not None else "N/A"
                ]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    with tab3:
        st.header("Data Visualizations")
        
        if st.session_state.processed_data is None:
            st.warning("Please upload and process data first.")
            return
        
        if st.session_state.analysis_results is None:
            st.warning("Please run analysis first.")
            return
        
        visualizer = WeatherVisualizer()
        data = st.session_state.processed_data
        results = st.session_state.analysis_results
        
        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization",
            [
                "Temperature Time Series",
                "Temperature Distribution",
                "Hot/Cold Days Calendar",
                "Temperature Trends",
                "Location-based Analysis",
                "Monthly Patterns"
            ]
        )
        
        try:
            if viz_option == "Temperature Time Series":
                fig = visualizer.create_temperature_timeseries(data, hot_threshold, cold_threshold)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Temperature Distribution":
                fig = visualizer.create_temperature_distribution(data, hot_threshold, cold_threshold)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Hot/Cold Days Calendar":
                fig = visualizer.create_calendar_heatmap(data, hot_threshold, cold_threshold)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Temperature Trends":
                fig = visualizer.create_temperature_trends(data)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Location-based Analysis":
                if 'location' in data.columns:
                    fig = visualizer.create_location_analysis(data, hot_threshold, cold_threshold)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Location data not available in the dataset.")
            
            elif viz_option == "Monthly Patterns":
                fig = visualizer.create_monthly_patterns(data, hot_threshold, cold_threshold)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
    with tab4:
        st.header("Analysis Summary")
        
        if st.session_state.analysis_results is None:
            st.warning("Please run analysis first.")
            return
        
        results = st.session_state.analysis_results
        data = st.session_state.processed_data
        
        # Summary metrics
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Hot Day Analysis")
            st.write(f"**Total Hot Days:** {results['hot_days_count']}")
            st.write(f"**Percentage:** {results['hot_days_percentage']:.1f}%")
            if results['hot_stats']['mean'] is not None:
                st.write(f"**Average Hot Day Temp:** {results['hot_stats']['mean']:.2f}°C")
                st.write(f"**Hottest Day:** {results['hot_stats']['max']:.2f}°C")
        
        with col2:
            st.markdown("### ❄️ Cold Day Analysis")
            st.write(f"**Total Cold Days:** {results['cold_days_count']}")
            st.write(f"**Percentage:** {results['cold_days_percentage']:.1f}%")
            if results['cold_stats']['mean'] is not None:
                st.write(f"**Average Cold Day Temp:** {results['cold_stats']['mean']:.2f}°C")
                st.write(f"**Coldest Day:** {results['cold_stats']['min']:.2f}°C")
        
        # Data period information
        st.subheader("Data Period")
        if 'timestamp' in data.columns:
            start_date = data['timestamp'].min()
            end_date = data['timestamp'].max()
            st.write(f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            st.write(f"**Duration:** {(end_date - start_date).days} days")
        
        # Export options
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Analysis Results"):
                # Create summary report
                report = {
                    'analysis_date': datetime.now().isoformat(),
                    'thresholds': {
                        'hot_threshold': hot_threshold,
                        'cold_threshold': cold_threshold
                    },
                    'results': results,
                    'data_summary': {
                        'total_records': len(data),
                        'date_range': {
                            'start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                            'end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
                        }
                    }
                }
                
                json_str = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"weather_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Download Processed Data"):
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV Data",
                    data=csv_data,
                    file_name=f"processed_weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
