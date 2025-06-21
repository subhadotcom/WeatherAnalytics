import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Union, List, Dict, Any
import os
import json

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def validate_date_range(start_date: Union[date, datetime], end_date: Union[date, datetime]) -> bool:
    """
    Validate that end date is after start date.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        True if valid date range, False otherwise
    """
    if start_date is None or end_date is None:
        return True
    
    return end_date >= start_date

def detect_data_format(file_content: bytes, filename: str) -> str:
    """
    Detect the format of uploaded data file.
    
    Args:
        file_content: Raw file content
        filename: Name of the file
        
    Returns:
        Detected format ('csv', 'json', 'log', 'unknown')
    """
    try:
        # First check file extension
        if filename.lower().endswith('.csv'):
            return 'csv'
        elif filename.lower().endswith('.json'):
            return 'json'
        elif filename.lower().endswith(('.log', '.txt')):
            return 'log'
        
        # Try to detect from content
        content_str = file_content.decode('utf-8', errors='ignore')[:1000]  # First 1000 chars
        
        # Check for JSON
        if content_str.strip().startswith(('{', '[')):
            try:
                json.loads(content_str[:500])
                return 'json'
            except:
                pass
        
        # Check for CSV (look for common delimiters)
        if ',' in content_str and '\n' in content_str:
            lines = content_str.split('\n')[:5]
            comma_count = sum(line.count(',') for line in lines)
            if comma_count > len(lines):  # More commas than lines suggests CSV
                return 'csv'
        
        # Default to log format for text files
        return 'log'
        
    except Exception:
        return 'unknown'

def clean_temperature_value(value: Any) -> float:
    """
    Clean and convert temperature value to float.
    
    Args:
        value: Raw temperature value
        
    Returns:
        Cleaned temperature as float, or NaN if invalid
    """
    if pd.isna(value):
        return np.nan
    
    try:
        # Convert to string first
        str_value = str(value).strip()
        
        # Remove common suffixes
        str_value = str_value.replace('°C', '').replace('°F', '').replace('C', '').replace('F', '')
        
        # Remove other non-numeric characters except minus and decimal point
        cleaned = ''.join(c for c in str_value if c.isdigit() or c in '.-')
        
        if not cleaned or cleaned == '.' or cleaned == '-':
            return np.nan
        
        temp = float(cleaned)
        
        # Basic validation - reasonable temperature range
        if -100 <= temp <= 100:
            return temp
        else:
            return np.nan
            
    except (ValueError, TypeError):
        return np.nan

def generate_summary_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for a column.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Dictionary with summary statistics
    """
    if column not in df.columns:
        return {}
    
    series = df[column].dropna()
    
    if series.empty:
        return {'count': 0, 'message': 'No valid data'}
    
    stats = {
        'count': len(series),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'range': float(series.max() - series.min()),
        'q25': float(series.quantile(0.25)),
        'q75': float(series.quantile(0.75)),
        'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
        'skewness': float(series.skew()),
        'kurtosis': float(series.kurtosis())
    }
    
    return stats

def classify_weather_conditions(temperature: float, hot_threshold: float, cold_threshold: float) -> str:
    """
    Classify weather condition based on temperature and thresholds.
    
    Args:
        temperature: Temperature value
        hot_threshold: Hot day threshold
        cold_threshold: Cold day threshold
        
    Returns:
        Weather classification string
    """
    if pd.isna(temperature):
        return 'Unknown'
    elif temperature > hot_threshold:
        return 'Hot'
    elif temperature < cold_threshold:
        return 'Cold'
    else:
        return 'Normal'

def calculate_temperature_percentiles(df: pd.DataFrame, column: str = 'temperature') -> Dict[str, float]:
    """
    Calculate temperature percentiles for analysis.
    
    Args:
        df: DataFrame with temperature data
        column: Temperature column name
        
    Returns:
        Dictionary with percentile values
    """
    if column not in df.columns:
        return {}
    
    series = df[column].dropna()
    
    if series.empty:
        return {}
    
    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        percentiles[f'p{p}'] = float(series.quantile(p/100))
    
    return percentiles

def detect_outliers(df: pd.DataFrame, column: str = 'temperature', method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in temperature data.
    
    Args:
        df: DataFrame with temperature data
        column: Temperature column name
        method: Outlier detection method ('iqr' or 'zscore')
        
    Returns:
        DataFrame with outlier flags
    """
    if column not in df.columns:
        return df
    
    df_copy = df.copy()
    series = df_copy[column].dropna()
    
    if series.empty:
        df_copy['is_outlier'] = False
        return df_copy
    
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df_copy['is_outlier'] = (df_copy[column] < lower_bound) | (df_copy[column] > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_indices = series[z_scores > 3].index
        df_copy['is_outlier'] = df_copy.index.isin(outlier_indices)
    
    else:
        df_copy['is_outlier'] = False
    
    return df_copy

def export_analysis_results(results: Dict[str, Any], filename: str = None) -> str:
    """
    Export analysis results to JSON format.
    
    Args:
        results: Analysis results dictionary
        filename: Optional filename for export
        
    Returns:
        JSON string of results
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    clean_results = convert_numpy_types(results)
    
    # Add metadata
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'analysis_results': clean_results
    }
    
    json_str = json.dumps(export_data, indent=2, default=str)
    
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write(json_str)
        except Exception as e:
            print(f"Error writing to file {filename}: {e}")
    
    return json_str

def validate_temperature_data(df: pd.DataFrame, column: str = 'temperature') -> Dict[str, Any]:
    """
    Validate temperature data quality.
    
    Args:
        df: DataFrame with temperature data
        column: Temperature column name
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_records': len(df),
        'valid_temperatures': 0,
        'invalid_temperatures': 0,
        'missing_temperatures': 0,
        'out_of_range_temperatures': 0,
        'temperature_range': None,
        'issues': []
    }
    
    if column not in df.columns:
        validation_results['issues'].append(f"Column '{column}' not found in data")
        return validation_results
    
    # Count valid, invalid, and missing temperatures
    series = df[column]
    validation_results['missing_temperatures'] = series.isna().sum()
    
    # Check numeric values
    numeric_mask = pd.to_numeric(series, errors='coerce').notna()
    validation_results['valid_temperatures'] = numeric_mask.sum()
    validation_results['invalid_temperatures'] = len(series) - validation_results['valid_temperatures'] - validation_results['missing_temperatures']
    
    # Check temperature range (reasonable values)
    if validation_results['valid_temperatures'] > 0:
        numeric_temps = pd.to_numeric(series, errors='coerce').dropna()
        
        validation_results['temperature_range'] = {
            'min': float(numeric_temps.min()),
            'max': float(numeric_temps.max())
        }
        
        # Count out-of-range temperatures (outside -60 to 70°C)
        out_of_range = ((numeric_temps < -60) | (numeric_temps > 70)).sum()
        validation_results['out_of_range_temperatures'] = int(out_of_range)
        
        # Add issues
        if validation_results['missing_temperatures'] > 0:
            validation_results['issues'].append(f"{validation_results['missing_temperatures']} missing temperature values")
        
        if validation_results['invalid_temperatures'] > 0:
            validation_results['issues'].append(f"{validation_results['invalid_temperatures']} invalid temperature values")
        
        if out_of_range > 0:
            validation_results['issues'].append(f"{out_of_range} temperatures outside reasonable range (-60°C to 70°C)")
    
    return validation_results

def get_data_completeness_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate data completeness report.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with completeness metrics
    """
    report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'column_completeness': {},
        'overall_completeness': 0
    }
    
    if len(df) == 0:
        return report
    
    # Calculate completeness for each column
    for column in df.columns:
        non_null_count = df[column].notna().sum()
        completeness_percentage = (non_null_count / len(df)) * 100
        
        report['column_completeness'][column] = {
            'non_null_count': int(non_null_count),
            'null_count': int(len(df) - non_null_count),
            'completeness_percentage': float(completeness_percentage)
        }
    
    # Calculate overall completeness
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.count().sum()
    report['overall_completeness'] = float((non_null_cells / total_cells) * 100) if total_cells > 0 else 0
    
    return report
