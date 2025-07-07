import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Smart Home Energy Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .day-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .appliance-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    
    .energy-tip {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">⚡ Smart Home Energy Calculator</h1>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("🏠 Home Configuration")
    
    # BHK input with info
    bhk = st.number_input(
        "Enter your BHK (Bedroom, Hall, Kitchen)",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of bedrooms + hall + kitchen"
    )
    
    st.markdown("---")
    st.header("📊 Calculation Method")
    
    # Show calculation formula
    base_energy = (bhk + 1) * 4 + (bhk + 1) * 8
    st.info(f"Base Energy: ({bhk}+1) × 4 + ({bhk}+1) × 8 = {base_energy} kWh")
    
    # Appliance energy consumption
    st.subheader("⚡ Appliance Consumption")
    appliance_data = {
        "Appliance": ["AC", "Fridge", "Washing Machine"],
        "Energy (kWh)": [3, 3, 3],
        "Icon": ["❄️", "🧊", "🧺"]
    }
    
    for i, (appliance, energy, icon) in enumerate(zip(
        appliance_data["Appliance"], 
        appliance_data["Energy (kWh)"], 
        appliance_data["Icon"]
    )):
        st.markdown(f"{icon} **{appliance}**: {energy} kWh")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📅 Weekly Energy Consumption")
    
    # Days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days_elec = {}
    
    # Create tabs for each day
    day_tabs = st.tabs(days)
    
    for i, day in enumerate(days):
        with day_tabs[i]:
            st.subheader(f"🗓️ {day}")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                ac = st.checkbox(f"❄️ AC", key=f"ac_{day}")
                
            with col_b:
                fridge = st.checkbox(f"🧊 Fridge", key=f"fridge_{day}")
                
            with col_c:
                washing_machine = st.checkbox(f"🧺 Washing Machine", key=f"wm_{day}")
            
            # Calculate energy for this day
            cal_energy = base_energy
            if ac:
                cal_energy += 3
            if fridge:
                cal_energy += 3
            if washing_machine:
                cal_energy += 3
            
            days_elec[day] = cal_energy
            
            # Show daily consumption
            st.markdown(f"""
            <div class="day-card">
                <h4>{day} Total: {cal_energy} kWh</h4>
                <p>Base: {base_energy} kWh + Appliances: {cal_energy - base_energy} kWh</p>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.header("📈 Quick Stats")
    
    # Calculate statistics
    total_weekly = sum(days_elec.values())
    avg_daily = total_weekly / 7
    max_day = max(days_elec, key=days_elec.get)
    min_day = min(days_elec, key=days_elec.get)
    
    # Display metrics
    st.metric("Total Weekly", f"{total_weekly:.1f} kWh")
    st.metric("Average Daily", f"{avg_daily:.1f} kWh")
    st.metric("Highest Day", f"{max_day}", f"{days_elec[max_day]:.1f} kWh")
    st.metric("Lowest Day", f"{min_day}", f"{days_elec[min_day]:.1f} kWh")
    
    # Energy cost estimation
    st.subheader("💰 Cost Estimation")
    rate_per_kwh = st.slider("Rate per kWh (₹)", 3.0, 10.0, 6.0, 0.5)
    weekly_cost = total_weekly * rate_per_kwh
    monthly_cost = weekly_cost * 4.33
    
    st.metric("Weekly Cost", f"₹{weekly_cost:.2f}")
    st.metric("Monthly Cost", f"₹{monthly_cost:.2f}")

# Visualization section
st.markdown("---")
st.header("📊 Energy Consumption Visualization")

col1, col2 = st.columns(2)

with col1:
    # Bar chart using Streamlit's built-in charting
    chart_df = pd.DataFrame({
        'Day': days,
        'Energy (kWh)': list(days_elec.values())
    })
    st.subheader("📊 Daily Energy Consumption")
    st.bar_chart(chart_df.set_index('Day'), use_container_width=True)

with col2:
    # Pie chart data using metrics instead
    if total_weekly > 0:
        base_total = base_energy * 7
        appliance_total = total_weekly - base_total
        
        st.subheader("🔌 Energy Breakdown")
        
        # Create columns for the breakdown
        break_col1, break_col2 = st.columns(2)
        
        with break_col1:
            st.metric("Base Consumption", f"{base_total:.1f} kWh", 
                     f"{(base_total/total_weekly)*100:.1f}%")
        
        with break_col2:
            st.metric("Appliances", f"{appliance_total:.1f} kWh", 
                     f"{(appliance_total/total_weekly)*100:.1f}%")
        
        # Visual progress bars
        st.write("**Weekly Distribution:**")
        base_percentage = base_total / total_weekly
        appliance_percentage = appliance_total / total_weekly
        
        st.write(f"Base: {base_percentage:.1%}")
        st.progress(base_percentage)
        st.write(f"Appliances: {appliance_percentage:.1%}")
        st.progress(appliance_percentage)

# Line chart for weekly trend using Streamlit's built-in line chart
st.subheader("📈 Weekly Energy Consumption Trend")
line_df = pd.DataFrame({
    'Day': days,
    'Energy (kWh)': list(days_elec.values())
})
st.line_chart(line_df.set_index('Day'), use_container_width=True)

# Data table
st.header("📋 Detailed Consumption Data")
df = pd.DataFrame({
    'Day': days,
    'Energy (kWh)': list(days_elec.values()),
    'Cost (₹)': [energy * rate_per_kwh for energy in days_elec.values()]
})

# Add color coding to the dataframe
def color_code_energy(val):
    if val > avg_daily:
        return 'background-color: #ffcccc'
    elif val < avg_daily * 0.8:
        return 'background-color: #ccffcc'
    else:
        return 'background-color: #ffffcc'

styled_df = df.style.applymap(color_code_energy, subset=['Energy (kWh)'])
st.dataframe(styled_df, use_container_width=True)

# Energy saving tips
st.markdown("---")
st.header("💡 Energy Saving Tips")

tips = [
    "🌡️ Set AC temperature to 24°C or higher to save up to 20% energy",
    "🧊 Keep fridge temperature between 37-40°F for optimal efficiency",
    "🧺 Use washing machine with full loads to maximize efficiency",
    "💡 Replace traditional bulbs with LED lights",
    "🔌 Unplug electronics when not in use to avoid phantom loads",
    "🌅 Use natural light during the day instead of artificial lighting"
]

for tip in tips:
    st.markdown(f"""
    <div class="energy-tip">
        {tip}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌱 Made with Streamlit for Smart Energy Management</p>
    <p>Calculate • Visualize • Optimize • Save</p>
</div>
""", unsafe_allow_html=True)