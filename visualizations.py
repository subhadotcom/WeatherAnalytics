import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

class WeatherVisualizer:
    """
    Creates interactive visualizations for weather data analysis.
    """
    
    def __init__(self):
        self.color_palette = {
            'hot': '#FF6B6B',
            'cold': '#4ECDC4',
            'normal': '#45B7D1',
            'background': '#F8F9FA',
            'text': '#2C3E50'
        }
    
    def create_temperature_timeseries(self, df: pd.DataFrame, hot_threshold: float, cold_threshold: float) -> go.Figure:
        """Create interactive temperature time series plot."""
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            # Create a simple scatter plot if no timestamps
            fig = px.scatter(
                df, 
                y='temperature', 
                title="Temperature Distribution (No Time Data)",
                labels={'temperature': 'Temperature (°C)', 'index': 'Record Index'}
            )
        else:
            # Filter data with valid timestamps
            df_time = df[df['timestamp'].notna()].copy()
            df_time = df_time.sort_values('timestamp')
            
            # Create base time series
            fig = px.line(
                df_time, 
                x='timestamp', 
                y='temperature',
                title="Temperature Time Series",
                labels={'timestamp': 'Date/Time', 'temperature': 'Temperature (°C)'}
            )
            
            # Color points by classification
            df_time['classification'] = df_time['temperature'].apply(
                lambda x: 'Hot' if x > hot_threshold else ('Cold' if x < cold_threshold else 'Normal')
            )
            
            # Add colored scatter points
            for classification, color in [('Hot', self.color_palette['hot']), 
                                        ('Cold', self.color_palette['cold']), 
                                        ('Normal', self.color_palette['normal'])]:
                data = df_time[df_time['classification'] == classification]
                if not data.empty:
                    fig.add_scatter(
                        x=data['timestamp'],
                        y=data['temperature'],
                        mode='markers',
                        name=f'{classification} Days',
                        marker=dict(color=color, size=4),
                        showlegend=True
                    )
        
        # Add threshold lines
        fig.add_hline(
            y=hot_threshold, 
            line_dash="dash", 
            line_color=self.color_palette['hot'],
            annotation_text=f"Hot Threshold ({hot_threshold}°C)"
        )
        
        fig.add_hline(
            y=cold_threshold, 
            line_dash="dash", 
            line_color=self.color_palette['cold'],
            annotation_text=f"Cold Threshold ({cold_threshold}°C)"
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def create_temperature_distribution(self, df: pd.DataFrame, hot_threshold: float, cold_threshold: float) -> go.Figure:
        """Create temperature distribution histogram with density curve."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Distribution', 'Box Plot by Classification'),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=df['temperature'],
                nbinsx=50,
                name='Temperature Distribution',
                marker_color=self.color_palette['normal'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add threshold lines to histogram
        fig.add_vline(
            x=hot_threshold,
            line_dash="dash",
            line_color=self.color_palette['hot'],
            annotation_text=f"Hot ({hot_threshold}°C)",
            row=1, col=1
        )
        
        fig.add_vline(
            x=cold_threshold,
            line_dash="dash",
            line_color=self.color_palette['cold'],
            annotation_text=f"Cold ({cold_threshold}°C)",
            row=1, col=1
        )
        
        # Box plots by classification
        df['classification'] = df['temperature'].apply(
            lambda x: 'Hot' if x > hot_threshold else ('Cold' if x < cold_threshold else 'Normal')
        )
        
        for classification, color in [('Hot', self.color_palette['hot']), 
                                    ('Cold', self.color_palette['cold']), 
                                    ('Normal', self.color_palette['normal'])]:
            data = df[df['classification'] == classification]['temperature']
            if not data.empty:
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=classification,
                        marker_color=color,
                        boxpoints='outliers'
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=600,
            title="Temperature Distribution Analysis",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        
        return fig
    
    def create_calendar_heatmap(self, df: pd.DataFrame, hot_threshold: float, cold_threshold: float) -> go.Figure:
        """Create calendar heatmap of temperature patterns."""
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            # Create a simple message plot if no timestamp data
            fig = go.Figure()
            fig.add_annotation(
                text="Calendar heatmap requires timestamp data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Calendar Heatmap - No Time Data Available",
                height=400
            )
            return fig
        
        # Filter data with valid timestamps
        df_time = df[df['timestamp'].notna()].copy()
        
        if df_time.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid timestamp data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create daily averages
        df_time['date'] = df_time['timestamp'].dt.date
        daily_data = df_time.groupby('date')['temperature'].mean().reset_index()
        daily_data['year'] = pd.to_datetime(daily_data['date']).dt.year
        daily_data['month'] = pd.to_datetime(daily_data['date']).dt.month
        daily_data['day'] = pd.to_datetime(daily_data['date']).dt.day
        daily_data['day_of_year'] = pd.to_datetime(daily_data['date']).dt.dayofyear
        daily_data['weekday'] = pd.to_datetime(daily_data['date']).dt.dayofweek
        
        # Classification
        daily_data['classification'] = daily_data['temperature'].apply(
            lambda x: 'Hot' if x > hot_threshold else ('Cold' if x < cold_threshold else 'Normal')
        )
        
        # Create heatmap data
        years = sorted(daily_data['year'].unique())
        
        if len(years) == 1:
            # Single year heatmap
            year_data = daily_data[daily_data['year'] == years[0]]
            
            # Create week-based heatmap
            year_data['week'] = pd.to_datetime(year_data['date']).dt.isocalendar().week
            
            fig = px.scatter(
                year_data,
                x='week',
                y='weekday',
                color='temperature',
                size_max=15,
                title=f"Temperature Calendar Heatmap - {years[0]}",
                labels={
                    'week': 'Week of Year',
                    'weekday': 'Day of Week',
                    'temperature': 'Temperature (°C)'
                },
                color_continuous_scale='RdYlBu_r'
            )
            
            fig.update_yaxes(
                tickmode='array',
                tickvals=list(range(7)),
                ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            )
            
        else:
            # Multi-year monthly heatmap
            monthly_data = daily_data.groupby(['year', 'month'])['temperature'].mean().reset_index()
            
            fig = px.imshow(
                monthly_data.pivot(index='year', columns='month', values='temperature'),
                title="Monthly Temperature Averages Heatmap",
                labels=dict(x="Month", y="Year", color="Temperature (°C)"),
                color_continuous_scale='RdYlBu_r'
            )
            
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=[calendar.month_abbr[i] for i in range(1, 13)]
            )
        
        fig.update_layout(height=500)
        return fig
    
    def create_temperature_trends(self, df: pd.DataFrame) -> go.Figure:
        """Create temperature trend analysis plot."""
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            # Create a simple trend plot using index
            fig = px.scatter(
                df.reset_index(), 
                x='index', 
                y='temperature',
                title="Temperature Trend (No Time Data)",
                trendline="ols"
            )
            return fig
        
        # Filter data with valid timestamps
        df_time = df[df['timestamp'].notna()].copy()
        df_time = df_time.sort_values('timestamp')
        
        if df_time.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid timestamp data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Create daily averages
        df_time['date'] = df_time['timestamp'].dt.date
        daily_data = df_time.groupby('date')['temperature'].agg(['mean', 'min', 'max']).reset_index()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Temperature Trends', 'Moving Averages'),
            vertical_spacing=0.1
        )
        
        # Daily min/max range
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['max'],
                fill=None,
                mode='lines',
                line_color='rgba(255,107,107,0.3)',
                name='Daily Max',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['min'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(78,205,196,0.3)',
                name='Daily Range',
                fillcolor='rgba(128,128,128,0.2)'
            ),
            row=1, col=1
        )
        
        # Daily average
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['mean'],
                mode='lines',
                name='Daily Average',
                line=dict(color=self.color_palette['normal'], width=2)
            ),
            row=1, col=1
        )
        
        # Moving averages
        if len(daily_data) >= 7:
            daily_data['ma7'] = daily_data['mean'].rolling(window=7).mean()
            fig.add_trace(
                go.Scatter(
                    x=daily_data['date'],
                    y=daily_data['ma7'],
                    mode='lines',
                    name='7-day MA',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        if len(daily_data) >= 30:
            daily_data['ma30'] = daily_data['mean'].rolling(window=30).mean()
            fig.add_trace(
                go.Scatter(
                    x=daily_data['date'],
                    y=daily_data['ma30'],
                    mode='lines',
                    name='30-day MA',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title="Temperature Trend Analysis",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        
        return fig
    
    def create_location_analysis(self, df: pd.DataFrame, hot_threshold: float, cold_threshold: float) -> go.Figure:
        """Create location-based temperature analysis."""
        if 'location' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No location data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Calculate statistics by location
        location_stats = []
        for location in df['location'].unique():
            location_data = df[df['location'] == location]
            hot_days = len(location_data[location_data['temperature'] > hot_threshold])
            cold_days = len(location_data[location_data['temperature'] < cold_threshold])
            
            location_stats.append({
                'location': location,
                'total_records': len(location_data),
                'avg_temp': location_data['temperature'].mean(),
                'min_temp': location_data['temperature'].min(),
                'max_temp': location_data['temperature'].max(),
                'hot_days': hot_days,
                'cold_days': cold_days,
                'hot_percentage': (hot_days / len(location_data)) * 100,
                'cold_percentage': (cold_days / len(location_data)) * 100
            })
        
        stats_df = pd.DataFrame(location_stats)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Temperature by Location',
                'Temperature Range by Location',
                'Hot Days Percentage',
                'Cold Days Percentage'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average temperature
        fig.add_trace(
            go.Bar(
                x=stats_df['location'],
                y=stats_df['avg_temp'],
                name='Avg Temp',
                marker_color=self.color_palette['normal']
            ),
            row=1, col=1
        )
        
        # Temperature range
        fig.add_trace(
            go.Scatter(
                x=stats_df['location'],
                y=stats_df['max_temp'],
                mode='markers',
                name='Max Temp',
                marker=dict(color=self.color_palette['hot'], size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=stats_df['location'],
                y=stats_df['min_temp'],
                mode='markers',
                name='Min Temp',
                marker=dict(color=self.color_palette['cold'], size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Hot days percentage
        fig.add_trace(
            go.Bar(
                x=stats_df['location'],
                y=stats_df['hot_percentage'],
                name='Hot Days %',
                marker_color=self.color_palette['hot']
            ),
            row=2, col=1
        )
        
        # Cold days percentage
        fig.add_trace(
            go.Bar(
                x=stats_df['location'],
                y=stats_df['cold_percentage'],
                name='Cold Days %',
                marker_color=self.color_palette['cold']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title="Location-based Temperature Analysis",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=2, col=2)
        
        return fig
    
    def create_monthly_patterns(self, df: pd.DataFrame, hot_threshold: float, cold_threshold: float) -> go.Figure:
        """Create monthly temperature pattern analysis."""
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            fig = go.Figure()
            fig.add_annotation(
                text="Monthly patterns require timestamp data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Filter data with valid timestamps
        df_time = df[df['timestamp'].notna()].copy()
        
        if df_time.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid timestamp data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        df_time['month'] = df_time['timestamp'].dt.month
        df_time['month_name'] = df_time['timestamp'].dt.strftime('%B')
        
        # Calculate monthly statistics
        monthly_stats = df_time.groupby(['month', 'month_name'])['temperature'].agg([
            'mean', 'min', 'max', 'std', 'count'
        ]).reset_index()
        
        # Calculate hot/cold days by month
        df_time['is_hot'] = df_time['temperature'] > hot_threshold
        df_time['is_cold'] = df_time['temperature'] < cold_threshold
        
        monthly_classification = df_time.groupby(['month', 'month_name']).agg({
            'is_hot': 'sum',
            'is_cold': 'sum',
            'temperature': 'count'
        }).reset_index()
        
        monthly_classification['hot_percentage'] = (monthly_classification['is_hot'] / monthly_classification['temperature']) * 100
        monthly_classification['cold_percentage'] = (monthly_classification['is_cold'] / monthly_classification['temperature']) * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Temperature Averages',
                'Monthly Temperature Range',
                'Hot/Cold Days by Month',
                'Monthly Temperature Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly averages
        fig.add_trace(
            go.Bar(
                x=monthly_stats['month_name'],
                y=monthly_stats['mean'],
                name='Average',
                marker_color=self.color_palette['normal']
            ),
            row=1, col=1
        )
        
        # Monthly range
        fig.add_trace(
            go.Scatter(
                x=monthly_stats['month_name'],
                y=monthly_stats['max'],
                mode='lines+markers',
                name='Max',
                line=dict(color=self.color_palette['hot']),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_stats['month_name'],
                y=monthly_stats['min'],
                mode='lines+markers',
                name='Min',
                line=dict(color=self.color_palette['cold']),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Hot/Cold days
        fig.add_trace(
            go.Bar(
                x=monthly_classification['month_name'],
                y=monthly_classification['hot_percentage'],
                name='Hot Days %',
                marker_color=self.color_palette['hot'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_classification['month_name'],
                y=monthly_classification['cold_percentage'],
                name='Cold Days %',
                marker_color=self.color_palette['cold'],
                yaxis='y2',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Box plot for distribution
        for month in sorted(df_time['month'].unique()):
            month_data = df_time[df_time['month'] == month]
            fig.add_trace(
                go.Box(
                    y=month_data['temperature'],
                    name=month_data['month_name'].iloc[0],
                    boxpoints=False,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            title="Monthly Temperature Patterns",
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
        
        return fig
