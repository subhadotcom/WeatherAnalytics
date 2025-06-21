import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class WeatherAnalyzer:
    """
    Analyzes weather data to identify hot and cold day patterns.
    """
    
    def __init__(self, hot_threshold: float = 30.0, cold_threshold: float = 5.0):
        """
        Initialize analyzer with temperature thresholds.
        
        Args:
            hot_threshold: Temperature above which a day is considered hot (°C)
            cold_threshold: Temperature below which a day is considered cold (°C)
        """
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of weather data.
        
        Args:
            df: Weather data DataFrame
            
        Returns:
            Dictionary containing analysis results
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
    
    def _calculate_temperature_stats(self, temperatures: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive temperature statistics."""
        if temperatures.empty:
            return self._empty_stats()
        
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
    
    def _empty_stats(self) -> Dict[str, None]:
        """Return empty statistics dictionary."""
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
        """Analyze temporal patterns in temperature data."""
        temporal_stats = {}
        
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            df_with_time = df[df['timestamp'].notna()].copy()
            
            if not df_with_time.empty:
                # Add time-based features
                df_with_time['hour'] = df_with_time['timestamp'].dt.hour
                df_with_time['day_of_week'] = df_with_time['timestamp'].dt.dayofweek
                df_with_time['month'] = df_with_time['timestamp'].dt.month
                df_with_time['season'] = df_with_time['month'].apply(self._get_season)
                
                # Hourly patterns
                hourly_stats = df_with_time.groupby('hour')['temperature'].agg(['mean', 'std', 'count']).to_dict('index')
                
                # Daily patterns
                daily_stats = df_with_time.groupby('day_of_week')['temperature'].agg(['mean', 'std', 'count']).to_dict('index')
                
                # Monthly patterns
                monthly_stats = df_with_time.groupby('month')['temperature'].agg(['mean', 'std', 'count']).to_dict('index')
                
                # Seasonal patterns
                seasonal_stats = df_with_time.groupby('season')['temperature'].agg(['mean', 'std', 'count']).to_dict('index')
                
                # Hot/Cold days by time periods
                df_with_time['is_hot'] = df_with_time['temperature'] > self.hot_threshold
                df_with_time['is_cold'] = df_with_time['temperature'] < self.cold_threshold
                
                temporal_stats = {
                    'hourly_patterns': hourly_stats,
                    'daily_patterns': daily_stats,
                    'monthly_patterns': monthly_stats,
                    'seasonal_patterns': seasonal_stats,
                    'hot_days_by_month': df_with_time.groupby('month')['is_hot'].sum().to_dict(),
                    'cold_days_by_month': df_with_time.groupby('month')['is_cold'].sum().to_dict(),
                    'hot_days_by_season': df_with_time.groupby('season')['is_hot'].sum().to_dict(),
                    'cold_days_by_season': df_with_time.groupby('season')['is_cold'].sum().to_dict(),
                }
        
        return temporal_stats
    
    def _analyze_by_location(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temperature patterns by location."""
        location_stats = {}
        
        if 'location' in df.columns:
            locations = df['location'].unique()
            
            for location in locations:
                location_data = df[df['location'] == location]
                
                hot_days = len(location_data[location_data['temperature'] > self.hot_threshold])
                cold_days = len(location_data[location_data['temperature'] < self.cold_threshold])
                
                location_stats[location] = {
                    'total_records': len(location_data),
                    'temperature_stats': self._calculate_temperature_stats(location_data['temperature']),
                    'hot_days': hot_days,
                    'cold_days': cold_days,
                    'hot_percentage': (hot_days / len(location_data)) * 100 if len(location_data) > 0 else 0,
                    'cold_percentage': (cold_days / len(location_data)) * 100 if len(location_data) > 0 else 0
                }
        
        return location_stats
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temperature trends over time."""
        trend_stats = {}
        
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            df_with_time = df[df['timestamp'].notna()].copy()
            
            if len(df_with_time) > 1:
                # Sort by timestamp
                df_with_time = df_with_time.sort_values('timestamp')
                
                # Calculate daily averages if we have multiple readings per day
                df_with_time['date'] = df_with_time['timestamp'].dt.date
                daily_temps = df_with_time.groupby('date')['temperature'].mean().reset_index()
                
                if len(daily_temps) > 1:
                    # Simple linear trend
                    x = np.arange(len(daily_temps))
                    y = daily_temps['temperature'].values
                    
                    # Calculate correlation coefficient as trend indicator
                    if len(x) > 1:
                        correlation = np.corrcoef(x, y)[0, 1]
                        trend_stats['linear_trend_correlation'] = float(correlation)
                        
                        # Temperature change over period
                        temp_change = y[-1] - y[0]
                        trend_stats['total_temperature_change'] = float(temp_change)
                        trend_stats['daily_average_change'] = float(temp_change / len(x)) if len(x) > 0 else 0
                
                # Moving averages
                if len(daily_temps) >= 7:
                    daily_temps['temp_ma7'] = daily_temps['temperature'].rolling(window=7).mean()
                    trend_stats['seven_day_moving_average'] = daily_temps['temp_ma7'].iloc[-1]
                
                if len(daily_temps) >= 30:
                    daily_temps['temp_ma30'] = daily_temps['temperature'].rolling(window=30).mean()
                    trend_stats['thirty_day_moving_average'] = daily_temps['temp_ma30'].iloc[-1]
        
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
