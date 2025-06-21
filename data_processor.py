import pandas as pd
import numpy as np
import json
import io
import re
from datetime import datetime
from typing import Union, List, Dict, Any

class WeatherDataProcessor:
    """
    Handles processing of various weather data formats including CSV, JSON, and text logs.
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'txt', 'log']
        
    def process_csv(self, file_content: bytes) -> pd.DataFrame:
        """
        Process CSV weather data files.
        
        Args:
            file_content: Raw file content as bytes
            
        Returns:
            Processed DataFrame with standardized columns
        """
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content_str = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode file with supported encodings")
            
            # Read CSV data
            df = pd.read_csv(io.StringIO(content_str))
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Parse timestamps
            df = self._parse_timestamps(df)
            
            # Clean temperature data
            df = self._clean_temperature_data(df)
            
            # Add metadata
            df['data_source'] = 'csv'
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def process_json(self, file_content: bytes) -> pd.DataFrame:
        """
        Process JSON weather data files.
        
        Args:
            file_content: Raw file content as bytes
            
        Returns:
            Processed DataFrame with standardized columns
        """
        try:
            content_str = file_content.decode('utf-8')
            
            # Handle different JSON structures
            try:
                # Try parsing as JSON array
                data = json.loads(content_str)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Handle nested JSON structures
                    df = self._flatten_json_dict(data)
                else:
                    raise ValueError("Unsupported JSON structure")
            except json.JSONDecodeError:
                # Try parsing as JSONL (JSON Lines)
                lines = content_str.strip().split('\n')
                records = []
                for line in lines:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
                
                if not records:
                    raise ValueError("No valid JSON records found")
                
                df = pd.DataFrame(records)
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            # Parse timestamps
            df = self._parse_timestamps(df)
            
            # Clean temperature data
            df = self._clean_temperature_data(df)
            
            # Add metadata
            df['data_source'] = 'json'
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing JSON file: {str(e)}")
    
    def process_text_log(self, file_content: bytes) -> pd.DataFrame:
        """
        Process text log weather data files.
        
        Args:
            file_content: Raw file content as bytes
            
        Returns:
            Processed DataFrame with standardized columns
        """
        try:
            content_str = file_content.decode('utf-8')
            lines = content_str.strip().split('\n')
            
            records = []
            
            # Common log patterns for weather data
            patterns = [
                # Pattern 1: timestamp temp location
                r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+temp:\s*(-?\d+\.?\d*)\s+(?:location:\s*(\w+))?',
                # Pattern 2: JSON-like in logs
                r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+\{.*"temperature":\s*(-?\d+\.?\d*).*\}',
                # Pattern 3: Simple space-separated
                r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(-?\d+\.?\d*)\s*(\w*)',
                # Pattern 4: Comma-separated log format
                r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s*(-?\d+\.?\d*),?\s*(\w*)',
            ]
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parsed = False
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        timestamp_str = match.group(1)
                        temperature = float(match.group(2))
                        location = match.group(3) if len(match.groups()) > 2 and match.group(3) else 'unknown'
                        
                        record = {
                            'timestamp_str': timestamp_str,
                            'temperature': temperature,
                            'location': location
                        }
                        records.append(record)
                        parsed = True
                        break
                
                # If no pattern matches, try to extract temperature values
                if not parsed:
                    temp_matches = re.findall(r'-?\d+\.?\d*', line)
                    if temp_matches:
                        # Assume first number is temperature
                        try:
                            temperature = float(temp_matches[0])
                            if -50 <= temperature <= 60:  # Reasonable temperature range
                                records.append({
                                    'timestamp_str': None,
                                    'temperature': temperature,
                                    'location': 'unknown'
                                })
                        except ValueError:
                            continue
            
            if not records:
                raise ValueError("No weather data found in log file")
            
            df = pd.DataFrame(records)
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            # Parse timestamps
            df = self._parse_timestamps(df)
            
            # Clean temperature data
            df = self._clean_temperature_data(df)
            
            # Add metadata
            df['data_source'] = 'log'
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing log file: {str(e)}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to common format."""
        column_mapping = {
            # Temperature columns
            'temp': 'temperature',
            'temperature_c': 'temperature',
            'temperature_celsius': 'temperature',
            'temp_c': 'temperature',
            'air_temp': 'temperature',
            'air_temperature': 'temperature',
            
            # Timestamp columns
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date_time': 'timestamp',
            'recorded_at': 'timestamp',
            'timestamp_str': 'timestamp',
            'date': 'timestamp',
            
            # Location columns
            'station': 'location',
            'station_id': 'location',
            'sensor_id': 'location',
            'site': 'location',
            'place': 'location',
            
            # Additional fields
            'humidity': 'humidity',
            'pressure': 'pressure',
            'wind_speed': 'wind_speed',
            'wind_direction': 'wind_direction'
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        if 'temperature' not in df.columns:
            raise ValueError("No temperature column found in data")
        
        # Add missing optional columns
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.NaT
        
        if 'location' not in df.columns:
            df['location'] = 'unknown'
        
        return df
    
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse various timestamp formats."""
        if 'timestamp' not in df.columns:
            return df
        
        # Common timestamp formats
        timestamp_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y'
        ]
        
        parsed_timestamps = []
        
        for timestamp in df['timestamp']:
            if pd.isna(timestamp) or timestamp is None:
                parsed_timestamps.append(pd.NaT)
                continue
            
            timestamp_str = str(timestamp).strip()
            parsed = False
            
            for fmt in timestamp_formats:
                try:
                    parsed_time = datetime.strptime(timestamp_str, fmt)
                    parsed_timestamps.append(parsed_time)
                    parsed = True
                    break
                except ValueError:
                    continue
            
            if not parsed:
                # Try pandas automatic parsing
                try:
                    parsed_time = pd.to_datetime(timestamp_str)
                    parsed_timestamps.append(parsed_time)
                except:
                    parsed_timestamps.append(pd.NaT)
        
        df['timestamp'] = parsed_timestamps
        return df
    
    def _clean_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate temperature data."""
        if 'temperature' not in df.columns:
            return df
        
        # Convert to numeric, handling various formats
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        
        # Remove outliers (temperatures outside reasonable range)
        df = df[(df['temperature'] >= -60) & (df['temperature'] <= 70)]
        
        # Add temperature categories
        df['temp_valid'] = df['temperature'].notna()
        
        return df
    
    def _flatten_json_dict(self, data: dict) -> pd.DataFrame:
        """Flatten nested JSON dictionary structure."""
        records = []
        
        def flatten_dict(obj, parent_key='', sep='_'):
            items = []
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                        else:
                            items.append((f"{new_key}_{i}", item))
                else:
                    items.append((new_key, v))
            return dict(items)
        
        if 'data' in data or 'records' in data or 'readings' in data:
            # Handle common nested structures
            key = 'data' if 'data' in data else ('records' if 'records' in data else 'readings')
            if isinstance(data[key], list):
                for record in data[key]:
                    flattened = flatten_dict(record) if isinstance(record, dict) else record
                    records.append(flattened)
            else:
                records.append(flatten_dict(data[key]))
        else:
            records.append(flatten_dict(data))
        
        return pd.DataFrame(records)
    
    def filter_by_date_range(self, df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        """Filter data by date range."""
        if 'timestamp' not in df.columns:
            return df
        
        mask = (df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))
        return df[mask].copy()
    
    def filter_by_temperature_range(self, df: pd.DataFrame, min_temp: float, max_temp: float) -> pd.DataFrame:
        """Filter data by temperature range."""
        if 'temperature' not in df.columns:
            return df
        
        mask = (df['temperature'] >= min_temp) & (df['temperature'] <= max_temp)
        return df[mask].copy()
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report."""
        report = {
            'total_records': len(df),
            'missing_temperature': df['temperature'].isna().sum(),
            'missing_timestamp': df['timestamp'].isna().sum() if 'timestamp' in df.columns else 0,
            'temperature_range': {
                'min': df['temperature'].min(),
                'max': df['temperature'].max(),
                'mean': df['temperature'].mean()
            } if not df['temperature'].isna().all() else None,
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            } if 'timestamp' in df.columns and not df['timestamp'].isna().all() else None,
            'unique_locations': df['location'].nunique() if 'location' in df.columns else 0
        }
        
        return report
