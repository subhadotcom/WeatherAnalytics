# Weather Data Analysis Application

## Overview

This is a Streamlit-based weather data analysis application designed to process and visualize weather sensor data, specifically focusing on identifying and analyzing hot and cold day patterns. The application provides interactive data visualization, statistical analysis, and supports multiple data formats including CSV, JSON, and text logs.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid UI development
- **Interactive Components**: Sidebar configuration, file uploads, data visualization widgets
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Wide layout with sidebar for configuration options

### Backend Architecture
- **Modular Design**: Separate modules for data processing, analysis, and visualization
- **Data Processing Pipeline**: Raw data → processed DataFrame → analysis results → visualizations
- **Error Handling**: Comprehensive validation and error handling throughout the pipeline

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and UI orchestration
- **Features**: 
  - File upload interface
  - Configuration sidebar with temperature thresholds
  - Session state management for processed data
  - Integration of all modules

### 2. Data Processor (`data_processor.py`)
- **Purpose**: Handle multiple data formats and standardize input
- **Features**:
  - Support for CSV, JSON, TXT, LOG formats
  - Multiple encoding support (UTF-8, Latin-1, CP1252)
  - Column name standardization
  - Timestamp parsing and temperature data cleaning

### 3. Weather Analyzer (`weather_analyzer.py`)
- **Purpose**: Core analysis logic for weather patterns
- **Features**:
  - Configurable hot/cold temperature thresholds
  - Statistical analysis (mean, median, standard deviation)
  - Day classification (hot, cold, normal)
  - Pattern identification and trend analysis

### 4. Visualizations (`visualizations.py`)
- **Purpose**: Create interactive charts and graphs
- **Features**:
  - Time series temperature plots
  - Statistical distribution charts
  - Color-coded temperature classifications
  - Responsive design with consistent color palette

### 5. Utilities (`utils.py`)
- **Purpose**: Common helper functions
- **Features**:
  - File size formatting
  - Date range validation
  - Data format detection

## Data Flow

1. **Data Input**: User uploads weather data files through Streamlit interface
2. **Format Detection**: System automatically detects file format (CSV, JSON, TXT, LOG)
3. **Data Processing**: Raw data is cleaned, standardized, and validated
4. **Analysis**: Temperature data is analyzed against configurable thresholds
5. **Visualization**: Results are displayed through interactive Plotly charts
6. **Session Management**: Processed data and results are cached in session state

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualization library
- **DateTime**: Date and time handling

### Data Processing
- **JSON**: Built-in JSON handling
- **IO**: File and stream operations
- **Re**: Regular expressions for text processing

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Deployment Target**: Autoscale deployment for production
- **Port Configuration**: Application runs on port 5000
- **Process Management**: Streamlit server with custom configuration

### Configuration Files
- **Streamlit Config**: Custom server settings in `.streamlit/config.toml`
- **Project Dependencies**: Managed through `pyproject.toml` with uv lock file
- **Environment**: Nix-based environment with stable channel

### Key Architectural Decisions

1. **Streamlit Choice**: Selected for rapid prototyping and built-in web interface capabilities
2. **Modular Design**: Separated concerns into distinct modules for maintainability and testing
3. **Plotly Integration**: Chosen for interactive visualizations and professional appearance
4. **Multi-format Support**: Designed to handle various weather data formats from different sensors
5. **Session State Management**: Implemented to avoid reprocessing data on UI interactions
6. **Configurable Thresholds**: Made temperature thresholds user-configurable for flexibility


### Deployment Commands
- Primary: `streamlit run app.py --server.port 8501 --server.address localhost`