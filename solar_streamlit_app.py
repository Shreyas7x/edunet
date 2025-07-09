import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Solar Panel Performance Analyzer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(45deg, #FFE066, #FF6B35);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FF6B35, #FFE066);
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B35, #FFE066);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 107, 53, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Feature ranges for data generation
feature_ranges = {
    'winter': {
        'irradiance': (300, 700),
        'humidity': (30, 70),
        'wind_speed': (1, 6),
        'ambient_temperature': (5, 20),
        'tilt_angle': (10, 40),
    },
    'summer': {
        'irradiance': (800, 1200),
        'humidity': (40, 80),
        'wind_speed': (2, 8),
        'ambient_temperature': (25, 45),
        'tilt_angle': (15, 35),
    },
    'monsoon': {
        'irradiance': (200, 600),
        'humidity': (70, 95),
        'wind_speed': (3, 10),
        'ambient_temperature': (20, 30),
        'tilt_angle': (20, 45),
    }
}

# Calculation functions
def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))

def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.22 * irradiance - 0.04 * humidity + 0.02 * wind_speed + 0.06 * ambient_temp - 0.015 * abs(tilt_angle - 25))

def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.15 * irradiance - 0.05 * humidity + 0.025 * wind_speed + 0.07 * ambient_temp - 0.025 * abs(tilt_angle - 35))

@st.cache_data
def generate_seasonal_data():
    """Generate comprehensive seasonal data"""
    
    # Winter data
    winter_months_days = {'November': 30, 'December': 31, 'January': 31, 'February': 28}
    winter_data = []
    for month, days in winter_months_days.items():
        for _ in range(days):
            irr = np.random.uniform(*feature_ranges['winter']['irradiance'])
            hum = np.random.uniform(*feature_ranges['winter']['humidity'])
            wind = np.random.uniform(*feature_ranges['winter']['wind_speed'])
            temp = np.random.uniform(*feature_ranges['winter']['ambient_temperature'])
            tilt = np.random.uniform(*feature_ranges['winter']['tilt_angle'])
            kwh = calc_kwh_winter(irr, hum, wind, temp, tilt)
            
            winter_data.append({
                'irradiance': round(irr, 2),
                'humidity': round(hum, 2),
                'wind_speed': round(wind, 2),
                'ambient_temperature': round(temp, 2),
                'tilt_angle': round(tilt, 2),
                'kwh': round(kwh, 2),
                'season': 'winter',
                'month': month
            })
    
    # Summer data
    summer_months_days = {'March': 31, 'April': 30, 'May': 31, 'June': 30}
    summer_data = []
    for month, days in summer_months_days.items():
        for _ in range(days):
            irr = np.random.uniform(*feature_ranges['summer']['irradiance'])
            hum = np.random.uniform(*feature_ranges['summer']['humidity'])
            wind = np.random.uniform(*feature_ranges['summer']['wind_speed'])
            temp = np.random.uniform(*feature_ranges['summer']['ambient_temperature'])
            tilt = np.random.uniform(*feature_ranges['summer']['tilt_angle'])
            kwh = calc_kwh_summer(irr, hum, wind, temp, tilt)
            
            summer_data.append({
                'irradiance': round(irr, 2),
                'humidity': round(hum, 2),
                'wind_speed': round(wind, 2),
                'ambient_temperature': round(temp, 2),
                'tilt_angle': round(tilt, 2),
                'kwh': round(kwh, 2),
                'season': 'summer',
                'month': month
            })
    
    # Monsoon data
    monsoon_months_days = {'July': 31, 'August': 31, 'September': 30, 'October': 31}
    monsoon_data = []
    for month, days in monsoon_months_days.items():
        for _ in range(days):
            irr = np.random.uniform(*feature_ranges['monsoon']['irradiance'])
            hum = np.random.uniform(*feature_ranges['monsoon']['humidity'])
            wind = np.random.uniform(*feature_ranges['monsoon']['wind_speed'])
            temp = np.random.uniform(*feature_ranges['monsoon']['ambient_temperature'])
            tilt = np.random.uniform(*feature_ranges['monsoon']['tilt_angle'])
            kwh = calc_kwh_monsoon(irr, hum, wind, temp, tilt)
            
            monsoon_data.append({
                'irradiance': round(irr, 2),
                'humidity': round(hum, 2),
                'wind_speed': round(wind, 2),
                'ambient_temperature': round(temp, 2),
                'tilt_angle': round(tilt, 2),
                'kwh': round(kwh, 2),
                'season': 'monsoon',
                'month': month
            })
    
    # Combine all seasons
    all_data = winter_data + summer_data + monsoon_data
    return pd.DataFrame(all_data)

# Main App
def main():
    st.markdown('<h1 class="main-header">☀️ Solar Panel Performance Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Generate data
    if st.sidebar.button("🔄 Generate New Dataset"):
        st.cache_data.clear()
    
    df = generate_seasonal_data()
    
    # Navigation
    page = st.sidebar.radio("📊 Navigate to:", [
        "📈 Data Overview",
        "🎯 Energy Prediction",
        "🌦️ Season Classification",
        "⚡ Interactive Predictor"
    ])
    
    # Data Overview Page
    if page == "📈 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card">Total Records<br><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card">Avg Energy Output<br><h2>{df["kwh"].mean():.2f} kWh</h2></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card">Max Energy<br><h2>{df["kwh"].max():.2f} kWh</h2></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-card">Seasons<br><h2>{df["season"].nunique()}</h2></div>', unsafe_allow_html=True)
        
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Energy Output by Season")
        fig = px.box(df, x='season', y='kwh', color='season',
                    title="Energy Output Distribution by Season")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🌡️ Feature Correlations")
        corr_matrix = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Monthly Energy Trends")
        monthly_avg = df.groupby('month')['kwh'].mean().reset_index()
        fig = px.bar(monthly_avg, x='month', y='kwh', color='kwh',
                    title="Average Energy Output by Month")
        st.plotly_chart(fig, use_container_width=True)
    
    # Energy Prediction Page
    elif page == "🎯 Energy Prediction":
        st.header("🎯 Energy Output Prediction (Linear Regression)")
        
        # Prepare data
        X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
        y = df['kwh']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="metric-card">R² Score<br><h2>{r2:.4f}</h2></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card">Mean Squared Error<br><h2>{mse:.4f}</h2></div>', unsafe_allow_html=True)
        
        # Model coefficients
        st.subheader("🔍 Model Coefficients")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        fig = px.bar(coef_df, x='Feature', y='Coefficient', color='Coefficient',
                    title="Feature Importance (Linear Regression Coefficients)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Predictions vs Actual
        st.subheader("📊 Predictions vs Actual")
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual kWh', 'y': 'Predicted kWh'},
                        title="Actual vs Predicted Energy Output")
        
        # Add diagonal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                     line=dict(color="red", width=2, dash="dash"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals,
                        labels={'x': 'Predicted kWh', 'y': 'Residuals'},
                        title="Residuals Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Season Classification Page
    elif page == "🌦️ Season Classification":
        st.header("🌦️ Season Classification (Logistic Regression)")
        
        # Prepare data
        X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']]
        y = df['season']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        st.markdown(f'<div class="metric-card">Classification Accuracy<br><h2>{accuracy:.4f}</h2></div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=le.classes_, y=le.classes_,
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Feature importance for classification
        st.subheader("🎯 Feature Importance")
        feature_importance = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Importance', color='Importance',
                    title="Feature Importance for Season Classification")
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive Predictor Page
    elif page == "⚡ Interactive Predictor":
        st.header("⚡ Interactive Energy Predictor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Input Parameters")
            
            irradiance = st.slider("☀️ Solar Irradiance (W/m²)", 200, 1200, 600)
            humidity = st.slider("💧 Humidity (%)", 20, 100, 50)
            wind_speed = st.slider("💨 Wind Speed (m/s)", 1, 10, 5)
            ambient_temp = st.slider("🌡️ Temperature (°C)", 0, 50, 25)
            tilt_angle = st.slider("📐 Panel Tilt Angle (°)", 0, 60, 30)
            
            season = st.selectbox("🌦️ Season", ['winter', 'summer', 'monsoon'])
        
        with col2:
            st.subheader("📊 Prediction Results")
            
            # Calculate energy based on season
            if season == 'winter':
                predicted_kwh = calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
            elif season == 'summer':
                predicted_kwh = calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
            else:  # monsoon
                predicted_kwh = calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
            
            st.markdown(f'<div class="metric-card">Predicted Energy Output<br><h2>{predicted_kwh:.2f} kWh</h2></div>', unsafe_allow_html=True)
            
            # Show seasonal context
            seasonal_data = df[df['season'] == season]
            seasonal_avg = seasonal_data['kwh'].mean()
            seasonal_max = seasonal_data['kwh'].max()
            
            st.markdown(f'<div class="metric-card">Seasonal Average<br><h2>{seasonal_avg:.2f} kWh</h2></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">Seasonal Maximum<br><h2>{seasonal_max:.2f} kWh</h2></div>', unsafe_allow_html=True)
            
            # Performance indicator
            performance = (predicted_kwh / seasonal_avg) * 100 if seasonal_avg > 0 else 0
            
            if performance > 110:
                st.success(f"🎉 Excellent! {performance:.1f}% of seasonal average")
            elif performance > 90:
                st.info(f"👍 Good! {performance:.1f}% of seasonal average")
            else:
                st.warning(f"⚠️ Below average: {performance:.1f}% of seasonal average")
        
        # Real-time parameter impact
        st.subheader("📈 Parameter Impact Analysis")
        
        # Create impact analysis
        base_params = [irradiance, humidity, wind_speed, ambient_temp, tilt_angle]
        param_names = ['Irradiance', 'Humidity', 'Wind Speed', 'Temperature', 'Tilt Angle']
        
        impact_data = []
        for i, param in enumerate(base_params):
            # Test +/- 20% change
            test_params = base_params.copy()
            
            # +20% change
            test_params[i] = param * 1.2
            if season == 'winter':
                kwh_high = calc_kwh_winter(*test_params)
            elif season == 'summer':
                kwh_high = calc_kwh_summer(*test_params)
            else:
                kwh_high = calc_kwh_monsoon(*test_params)
            
            # -20% change
            test_params[i] = param * 0.8
            if season == 'winter':
                kwh_low = calc_kwh_winter(*test_params)
            elif season == 'summer':
                kwh_low = calc_kwh_summer(*test_params)
            else:
                kwh_low = calc_kwh_monsoon(*test_params)
            
            impact_data.append({
                'Parameter': param_names[i],
                'Impact': (kwh_high - kwh_low) / 2,
                'Positive_Change': kwh_high - predicted_kwh,
                'Negative_Change': predicted_kwh - kwh_low
            })
        
        impact_df = pd.DataFrame(impact_data)
        fig = px.bar(impact_df, x='Parameter', y='Impact', color='Impact',
                    title="Parameter Sensitivity Analysis (+/- 20% change)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
