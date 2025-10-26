# Standard library imports
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class WeatherAnalyzer:
    """
    A comprehensive analyzer for weather data that identifies patterns in temperature data.
    
    This class provides methods to analyze temperature data, identify hot and cold days,
    and extract various statistical insights. It supports both basic statistical analysis
    and more complex temporal and spatial pattern detection.
    
    Attributes:
        hot_threshold (float): Temperature threshold for hot days in °C
        cold_threshold (float): Temperature threshold for cold days in °C
    """
    
    def __init__(self, hot_threshold: float = 30.0, cold_threshold: float = 5.0) -> None:
        """
        Initialize the WeatherAnalyzer with temperature thresholds.
        
        Args:
            hot_threshold: Temperature in °C above which a day is considered hot.
                          Defaults to 30.0°C.
            cold_threshold: Temperature in °C below which a day is considered cold.
                          Defaults to 5.0°C.
                          
        Note:
            The hot_threshold should be greater than cold_threshold for meaningful analysis.
        """
        if hot_threshold <= cold_threshold:
            warnings.warn(
                f"hot_threshold ({hot_threshold}°C) should be greater than "
                f"cold_threshold ({cold_threshold}°C) for meaningful analysis."
            )
            
        self.hot_threshold = float(hot_threshold)
        self.cold_threshold = float(cold_threshold)
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of weather data.
        
        This is the main method that orchestrates the analysis of weather data.
        It performs the following analyses:
        - Basic temperature statistics
        - Classification of days into hot, cold, and normal
        - Temporal pattern analysis
        - Location-based analysis (if location data is available)
        - Trend analysis over time
        - Identification of extreme events
        - Analysis of temperature streaks
        
        Args:
            df: DataFrame containing weather data. Must include a 'temperature' column.
                Optional columns include 'timestamp' for temporal analysis and 'location'
                for spatial analysis.
                
        Returns:
            Dict[str, Any]: A dictionary containing comprehensive analysis results with
                          the following structure:
                          {
                              'total_records': int,
                              'hot_days_count': int,
                              'cold_days_count': int,
                              'normal_days_count': int,
                              'hot_days_percentage': float,
                              'cold_days_percentage': float,
                              'normal_days_percentage': float,
                              'temp_stats': Dict[str, float],
                              'hot_stats': Dict[str, float],
                              'cold_stats': Dict[str, float],
                              'temporal_analysis': Dict[str, Any],
                              'location_analysis': Optional[Dict[str, Any]],
                              'trend_analysis': Dict[str, Any],
                              'extreme_events': Dict[str, Any],
                              'streaks': Dict[str, Any]
                          }
                          
        Raises:
            ValueError: If the input DataFrame is empty or missing required columns.
        """
        if df.empty:
            raise ValueError("No data provided for analysis")
        
        if 'temperature' not in df.columns:
            raise ValueError("Temperature column not found in data")
        
        # Remove invalid temperature readings
        valid_data = df[df['temperature'].notna()].copy()
        
        if valid_data.empty:
            raise ValueError("No valid temperature data found")
        
        # Classify days
        hot_days = valid_data[valid_data['temperature'] > self.hot_threshold]
        cold_days = valid_data[valid_data['temperature'] < self.cold_threshold]
        normal_days = valid_data[
            (valid_data['temperature'] >= self.cold_threshold) & 
            (valid_data['temperature'] <= self.hot_threshold)
        ]
        
        # Calculate basic statistics
        results = {
            'total_records': len(valid_data),
            'hot_days_count': len(hot_days),
            'cold_days_count': len(cold_days),
            'normal_days_count': len(normal_days),
            'hot_days_percentage': (len(hot_days) / len(valid_data)) * 100,
            'cold_days_percentage': (len(cold_days) / len(valid_data)) * 100,
            'normal_days_percentage': (len(normal_days) / len(valid_data)) * 100,
            
            # Overall temperature statistics
            'temp_stats': self._calculate_temperature_stats(valid_data['temperature']),
            
            # Hot days statistics
            'hot_stats': self._calculate_temperature_stats(hot_days['temperature']) if not hot_days.empty else self._empty_stats(),
            
            # Cold days statistics
            'cold_stats': self._calculate_temperature_stats(cold_days['temperature']) if not cold_days.empty else self._empty_stats(),
            
            # Temporal analysis
            'temporal_analysis': self._analyze_temporal_patterns(valid_data),
            
            # Location analysis (if available)
            'location_analysis': self._analyze_by_location(valid_data) if 'location' in valid_data.columns else None,
            
            # Trend analysis
            'trend_analysis': self._analyze_trends(valid_data),
            
            # Extreme events
            'extreme_events': self._identify_extreme_events(valid_data),
            
            # Streaks analysis
            'streaks': self._analyze_streaks(valid_data)
        }
        
        return results
    
    def _calculate_temperature_stats(self, temperatures: pd.Series) -> Dict[str, Optional[float]]:
        """
        Calculate comprehensive statistics for a temperature series.
        
        Args:
            temperatures: Pandas Series containing temperature values
            
        Returns:
            Dict[str, Optional[float]]: Dictionary containing the following statistics:
                - count: Number of non-null values
                - mean: Average temperature
                - median: Median temperature
                - std: Standard deviation of temperatures
                - min: Minimum temperature
                - max: Maximum temperature
                - q25: 25th percentile
                - q75: 75th percentile
                - range: Difference between max and min temperatures
                
            Returns all values as None if the input series is empty.
        """
        if temperatures.empty:
            return self._empty_stats()
        
        try:
            return {
                'count': len(temperatures),
                'mean': float(temperatures.mean()),
                'median': float(temperatures.median()),
                'std': float(temperatures.std()),
                'min': float(temperatures.min()),
                'max': float(temperatures.max()),
                'q25': float(temperatures.quantile(0.25)),
                'q75': float(temperatures.quantile(0.75)),
                'range': float(temperatures.max() - temperatures.min())
            }
        except Exception as e:
            warnings.warn(f"Error calculating temperature statistics: {str(e)}")
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict[str, None]:
        """
        Return a dictionary with all statistics set to None.
        
        This is used as a fallback when statistics cannot be calculated.
        
        Returns:
            Dict[str, None]: Dictionary with all statistic values set to None
        """
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None,
            'range': None
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in temperature data.
        
        This method extracts various temporal patterns from the temperature data,
        including hourly, daily, monthly, and seasonal variations.
        
        Args:
            df: DataFrame containing 'timestamp' and 'temperature' columns
            
        Returns:
            Dict[str, Any]: Dictionary containing temporal analysis results with the following structure:
                {
                    'hourly_patterns': Dict[int, Dict[str, float]],  # Stats by hour of day
                    'daily_patterns': Dict[int, Dict[str, float]],   # Stats by day of week
                    'monthly_patterns': Dict[int, Dict[str, float]], # Stats by month
                    'seasonal_patterns': Dict[str, Dict[str, float]], # Stats by season
                    'hot_days_by_month': Dict[int, int],  # Count of hot days by month
                    'cold_days_by_month': Dict[int, int], # Count of cold days by month
                    'hot_days_by_season': Dict[str, int], # Count of hot days by season
                    'cold_days_by_season': Dict[str, int] # Count of cold days by season
                }
                
            Returns an empty dict if no timestamp data is available.
        """
        temporal_stats = {}
        
        # Only proceed if we have timestamp data
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            return temporal_stats
            
        try:
            # Filter out rows with missing timestamps and create a working copy
            df_with_time = df[df['timestamp'].notna()].copy()
            
            if df_with_time.empty:
                return temporal_stats
                
            # Extract time-based features
            df_with_time['hour'] = df_with_time['timestamp'].dt.hour
            df_with_time['day_of_week'] = df_with_time['timestamp'].dt.dayofweek
            df_with_time['month'] = df_with_time['timestamp'].dt.month
            df_with_time['season'] = df_with_time['month'].apply(self._get_season)
            
            # Calculate statistics for different time periods
            time_periods = {
                'hourly': 'hour',
                'daily': 'day_of_week',
                'monthly': 'month',
                'seasonal': 'season'
            }
            
            # Initialize result dictionary
            temporal_stats = {f'{period}_patterns': {} for period in time_periods}
            
            # Add statistics for each time period
            for period_name, period_col in time_periods.items():
                stats = df_with_time.groupby(period_col)['temperature'] \
                                  .agg(['mean', 'std', 'count']) \
                                  .to_dict('index')
                temporal_stats[f'{period_name}_patterns'] = stats
            
            # Calculate hot/cold day counts by time periods
            df_with_time['is_hot'] = df_with_time['temperature'] > self.hot_threshold
            df_with_time['is_cold'] = df_with_time['temperature'] < self.cold_threshold
            
            # Add hot/cold day counts to results
            temporal_stats.update({
                'hot_days_by_month': df_with_time.groupby('month')['is_hot'].sum().to_dict(),
                'cold_days_by_month': df_with_time.groupby('month')['is_cold'].sum().to_dict(),
                'hot_days_by_season': df_with_time.groupby('season')['is_hot'].sum().to_dict(),
                'cold_days_by_season': df_with_time.groupby('season')['is_cold'].sum().to_dict(),
            })
            
        except Exception as e:
            warnings.warn(f"Error in temporal pattern analysis: {str(e)}")
            
        return temporal_stats
        
        return temporal_stats
    
    def _analyze_by_location(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temperature patterns grouped by location.
        
        This method groups temperature data by location and calculates statistics
        for each location, including hot/cold day counts and percentages.
        
        Args:
            df: DataFrame containing 'location' and 'temperature' columns
            
        Returns:
            Dict[str, Dict]: A dictionary where keys are location names and values
                           are dictionaries containing location-specific statistics:
                           {
                               'total_records': int,
                               'temperature_stats': Dict[str, float],
                               'hot_days': int,
                               'cold_days': int,
                               'hot_percentage': float,
                               'cold_percentage': float
                           }
                           
            Returns an empty dict if no location data is available.
        """
        location_stats = {}
        
        # Only proceed if location data is available
        if 'location' not in df.columns or df['location'].isna().all():
            return location_stats
            
        try:
            # Get unique locations, excluding NaN values
            locations = df['location'].dropna().unique()
            
            if len(locations) == 0:
                return location_stats
                
            # Process each location
            for location in locations:
                # Filter data for this location
                location_data = df[df['location'] == location]
                
                # Skip if no data for this location
                if location_data.empty:
                    continue
                    
                # Calculate hot/cold day counts
                hot_days = len(location_data[location_data['temperature'] > self.hot_threshold])
                cold_days = len(location_data[location_data['temperature'] < self.cold_threshold])
                total_days = len(location_data)
                
                # Calculate statistics for this location
                location_stats[str(location)] = {
                    'total_records': total_days,
                    'temperature_stats': self._calculate_temperature_stats(location_data['temperature']),
                    'hot_days': hot_days,
                    'cold_days': cold_days,
                    'hot_percentage': (hot_days / total_days) * 100 if total_days > 0 else 0.0,
                    'cold_percentage': (cold_days / total_days) * 100 if total_days > 0 else 0.0
                }
                
        except Exception as e:
            warnings.warn(f"Error in location analysis: {str(e)}")
            
        return location_stats
        
        return location_stats
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temperature trends over time.
        
        This method identifies and quantifies trends in temperature data over time,
        including linear trends and seasonal patterns.
        
        Args:
            df: DataFrame containing 'timestamp' and 'temperature' columns
            
        Returns:
            Dict[str, Any]: Dictionary containing trend analysis results with the following structure:
                {
                    'linear_trend': {
                        'slope': float,  # Slope of the trend line (°C/day)
                        'intercept': float,  # Y-intercept of the trend line
                        'r_value': float,  # Correlation coefficient (-1 to 1)
                        'p_value': float,  # Statistical significance of the trend
                        'std_err': float   # Standard error of the estimate
                    },
                    'seasonal_strength': float,  # Strength of seasonal patterns (0-1)
                    'trend_strength': float,     # Strength of the trend (0-1)
                    'residuals': List[float],     # Residuals from the trend line
                    'is_significant': bool        # Whether the trend is statistically significant (p < 0.05)
                }
                
            Returns an empty dict if insufficient data is available for trend analysis.
        """
        trend_stats = {}
        
        # Check if we have timestamp data
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            return trend_stats
            
        try:
            # Filter out rows with missing timestamps and sort
            df_with_time = df[df['timestamp'].notna()].copy()
            
            # Need at least 2 points for trend analysis
            if len(df_with_time) < 2:
                return trend_stats
                
            # Sort by timestamp to ensure correct ordering
            df_with_time = df_with_time.sort_values('timestamp')
            
            # Convert timestamps to numerical values for regression
            # (number of days since the first timestamp)
            min_time = df_with_time['timestamp'].min()
            df_with_time['days_since_start'] = (
                (df_with_time['timestamp'] - min_time).dt.total_seconds() / (24 * 3600)
            )
            
            # Calculate daily averages if we have multiple readings per day
            df_with_time['date'] = df_with_time['timestamp'].dt.date
            daily_temps = df_with_time.groupby('date')['temperature'].mean().reset_index()
            
            # Convert dates to numerical values
            min_date = daily_temps['date'].min()
            daily_temps['days'] = (daily_temps['date'] - min_date).dt.days
            
            # Skip if we don't have enough data points
            if len(daily_temps) < 2:
                return trend_stats
                
            # Calculate linear regression
            from scipy.stats import linregress
            try:
                slope, intercept, r_value, p_value, std_err = linregress(
                    daily_temps['days'], 
                    daily_temps['temperature']
                )
                
                # Calculate residuals
                predicted = slope * daily_temps['days'] + intercept
                residuals = daily_temps['temperature'] - predicted
                
                # Calculate trend and seasonal strength
                temp_variance = np.var(daily_temps['temperature'])
                trend_strength = max(0, min(1, 1 - (np.var(residuals) / temp_variance))) if temp_variance > 0 else 0.0
                
                # Simple seasonal decomposition (additive model)
                daily_temps['moving_avg'] = daily_temps['temperature'].rolling(window=min(30, len(daily_temps)), min_periods=1).mean()
                detrended = daily_temps['temperature'] - daily_temps['moving_avg']
                
                # Simple seasonal strength (variance ratio)
                detrended_variance = np.var(detrended)
                seasonal_strength = 0.0
                if detrended_variance > 0:
                    seasonal_strength = max(0, min(1, 1 - (np.var(residuals - detrended) / detrended_variance)))
                
                trend_stats = {
                    'linear_trend': {
                        'slope': float(slope * 365.25),  # Convert to °C/year
                        'intercept': float(intercept),
                        'r_value': float(r_value),
                        'p_value': float(p_value),
                        'std_err': float(std_err)
                    },
                    'seasonal_strength': float(seasonal_strength),
                    'trend_strength': float(trend_strength),
                    'residuals': residuals.tolist(),
                    'is_significant': p_value < 0.05 if p_value is not None else False
                }
                
            except (ValueError, np.linalg.LinAlgError) as e:
                warnings.warn(f"Error in trend analysis: {str(e)}")
                return trend_stats
                
        except Exception as e:
            warnings.warn(f"Error processing trend data: {str(e)}")
            return trend_stats
            
        return trend_stats
    
    def _identify_extreme_events(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify extreme temperature events."""
        extreme_events = {}
        
        if not df.empty:
            temps = df['temperature']
            
            # Statistical outliers (using IQR method)
            q1 = temps.quantile(0.25)
            q3 = temps.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df['temperature'] < lower_bound) | (df['temperature'] > upper_bound)]
            
            extreme_events = {
                'total_outliers': len(outliers),
                'extreme_hot_events': len(df[df['temperature'] > upper_bound]),
                'extreme_cold_events': len(df[df['temperature'] < lower_bound]),
                'hottest_temperature': float(temps.max()),
                'coldest_temperature': float(temps.min()),
                'temperature_range': float(temps.max() - temps.min())
            }
            
            # Add timestamp information if available
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                hottest_idx = temps.idxmax()
                coldest_idx = temps.idxmin()
                
                if not pd.isna(df.loc[hottest_idx, 'timestamp']):
                    extreme_events['hottest_date'] = df.loc[hottest_idx, 'timestamp'].isoformat()
                
                if not pd.isna(df.loc[coldest_idx, 'timestamp']):
                    extreme_events['coldest_date'] = df.loc[coldest_idx, 'timestamp'].isoformat()
        
        return extreme_events
    
    def _analyze_streaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consecutive hot and cold day streaks."""
        streaks = {}
        
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            df_sorted = df[df['timestamp'].notna()].sort_values('timestamp').copy()
            
            if not df_sorted.empty:
                # Create daily temperature data
                df_sorted['date'] = df_sorted['timestamp'].dt.date
                daily_temps = df_sorted.groupby('date')['temperature'].agg(['mean', 'max', 'min']).reset_index()
                
                # Classify days
                daily_temps['is_hot'] = daily_temps['mean'] > self.hot_threshold
                daily_temps['is_cold'] = daily_temps['mean'] < self.cold_threshold
                
                # Calculate streaks
                hot_streaks = self._calculate_consecutive_streaks(daily_temps['is_hot'])
                cold_streaks = self._calculate_consecutive_streaks(daily_temps['is_cold'])
                
                streaks = {
                    'longest_hot_streak': max(hot_streaks) if hot_streaks else 0,
                    'average_hot_streak': np.mean(hot_streaks) if hot_streaks else 0,
                    'total_hot_streaks': len(hot_streaks),
                    'longest_cold_streak': max(cold_streaks) if cold_streaks else 0,
                    'average_cold_streak': np.mean(cold_streaks) if cold_streaks else 0,
                    'total_cold_streaks': len(cold_streaks)
                }
        
        return streaks
    
    def _calculate_consecutive_streaks(self, boolean_series: pd.Series) -> List[int]:
        """Calculate lengths of consecutive True values in a boolean series."""
        streaks = []
        current_streak = 0
        
        for value in boolean_series:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0
        
        # Don't forget the last streak if it ends with True
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks
    
    def _get_season(self, month: int) -> str:
        """Get season name from month number."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def update_thresholds(self, hot_threshold: float, cold_threshold: float):
        """Update temperature thresholds."""
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold
    
    def get_daily_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get daily temperature classifications."""
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            df['classification'] = df['temperature'].apply(self._classify_temperature)
            return df
        
        df_with_time = df[df['timestamp'].notna()].copy()
        df_with_time['date'] = df_with_time['timestamp'].dt.date
        
        # Calculate daily averages
        daily_temps = df_with_time.groupby('date')['temperature'].mean().reset_index()
        daily_temps['classification'] = daily_temps['temperature'].apply(self._classify_temperature)
        
        return daily_temps
    
    def _classify_temperature(self, temp: float) -> str:
        """Classify individual temperature reading."""
        if pd.isna(temp):
            return 'Unknown'
        elif temp > self.hot_threshold:
            return 'Hot'
        elif temp < self.cold_threshold:
            return 'Cold'
        else:
            return 'Normal'
