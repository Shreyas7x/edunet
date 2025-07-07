import streamlit as st
import random

# Set page configuration
st.set_page_config(
    page_title="SIGMA ENERGY CALCULATOR 💀🔥",
    page_icon="💀",
    layout="centered"
)

# Brainrot CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        animation: gradient 15s ease infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff0080, #8000ff);
        color: white;
        border: none;
        font-weight: bold;
        text-transform: uppercase;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Brainrot phrases
brainrot_phrases = [
    "💀 ABSOLUTELY SENDING ME FR FR 💀",
    "🔥 THIS IS SO FIRE NO CAP 🔥",
    "💯 PERIODT BESTIE 💯",
    "✨ SLAY QUEEN ENERGY ✨",
    "🚫🧢 NO CAP DETECTED 🚫🧢",
    "📱 VERY MINDFUL VERY DEMURE 📱",
    "🎭 OHIO ENERGY CALCULATOR 🎭",
    "👑 SIGMA GRINDSET ACTIVATED 👑"
]

# Title with random brainrot
st.title("💀🔥 SIGMA ENERGY CALCULATOR 🔥💀")
st.markdown(f"### {random.choice(brainrot_phrases)}")
st.markdown("**Calculate your house's energy consumption but make it ✨AESTHETIC✨**")

# Animated separator
st.markdown("---")
st.markdown("### 🎭 FILL OUT THIS FORM OR YOU'RE GIVING OHIO ENERGY 🎭")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 WHO ARE YOU BBG?")
    
    # Name input with brainrot placeholder
    name = st.text_input("Your Government Name 📛", placeholder="Enter your name bestie")
    
    # Age input with validation
    age = st.number_input("How Many Years You Been Slaying? 🎂", min_value=1, max_value=100, value=25, step=1)
    
    # City input
    city = st.text_input("What City You Reppin? 🏙️", placeholder="Drop your city queen")
    
    # Area input
    area = st.text_input("Your Neighborhood (No Doxxing) 🏘️", placeholder="Enter your area")

with col2:
    st.subheader("🏠 HOUSE SPECS (VERY MINDFUL)")
    
    # Housing type with brainrot options
    housing_type = st.selectbox(
        "Are You Living in a Flat or Tenement? 🏠",
        ["Flat (Basic)", "Tenement (Bougie)"]
    )
    
    # BHK selection with brainrot descriptions
    bhk_options = {
        1: "1 BHK (Single But Not Ready To Mingle)",
        2: "2 BHK (Couple Goals)",
        3: "3 BHK (Family Vibes)"
    }
    
    bhk_display = st.selectbox(
        "How Many Rooms We Working With? 🛏️",
        list(bhk_options.values())
    )
    
    # Extract actual BHK number
    bhk = int(bhk_display.split()[0])
    
    st.subheader("🔌 WHAT APPLIANCES YOU GOT?")
    
    # Appliance inputs with brainrot descriptions
    ac = st.number_input("Air Conditioners (Cool Kid Equipment) ❄️", min_value=0, value=0, step=1)
    fridge = st.number_input("Refrigerators (Food Storage Slay) 🧊", min_value=0, value=1, step=1)
    wm = st.number_input("Washing Machines (Clean Clothes Era) 🧺", min_value=0, value=0, step=1)

# Calculate button with extra brainrot
if st.button("CALCULATE MY ENERGY CONSUMPTION FR FR 💀🔥", type="primary"):
    if name and city and area:
        # Calculate base energy based on BHK
        base_energy = {1: 2.4, 2: 3.6, 3: 4.8}
        total_energy = base_energy[bhk]
        
        # Add appliance energy consumption
        total_energy += (ac * 3) + (fridge * 4) + (wm * 3)
        
        # Random success message
        success_messages = [
            "✅ CALCULATION COMPLETE BESTIE!",
            "✅ PERIODT! YOUR RESULTS ARE READY!",
            "✅ SLAY! WE GOT YOUR NUMBERS!",
            "✅ NO CAP, HERE'S YOUR ENERGY TEA!"
        ]
        st.success(random.choice(success_messages))
        
        # Create results section
        st.subheader("📊 THE TEA ON YOUR ENERGY CONSUMPTION")
        
        # Display user info in an expander with brainrot
        with st.expander("👤 Your Main Character Info"):
            st.write(f"**Name:** {name} 💅")
            st.write(f"**Age:** {age} years of pure slay 🎂")
            st.write(f"**Location:** {area}, {city} 📍")
            st.write(f"**Housing:** {bhk} BHK {housing_type.split()[0]} (very mindful, very demure) 🏠")
        
        # Energy breakdown with brainrot metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Base Energy (Your Foundation) 🏠", f"{base_energy[bhk]} kWh")
            st.metric("AC Energy (Staying Cool) ❄️", f"{ac * 3} kWh")
        
        with col2:
            st.metric("Fridge Energy (Keeping It Fresh) 🧊", f"{fridge * 4} kWh")
            st.metric("Washing Machine Energy (Clean Era) 🧺", f"{wm * 3} kWh")
        
        # Total energy consumption with brainrot
        st.metric(
            "🔋 TOTAL ENERGY CONSUMPTION (THE MAIN CHARACTER MOMENT)", 
            f"{total_energy} kWh",
            delta=f"{total_energy - base_energy[bhk]} kWh from your appliances bestie"
        )
        
        # Additional insights with maximum brainrot
        st.subheader("💡 ENERGY INSIGHTS (SPILLING THE TEA)")
        
        if total_energy > 10:
            st.warning("⚠️ BESTIE YOU'RE USING TOO MUCH ENERGY! This is giving wasteful energy, not sustainable queen behavior 💀")
        elif total_energy > 5:
            st.info("ℹ️ Your energy consumption is giving balanced vibes! Very mindful, very demure 💅")
        else:
            st.success("🌱 PERIODT! Your energy consumption is absolutely sending me! Eco-friendly queen behavior! 👑")
        
        # Random motivational brainrot
        motivation = [
            "Keep slaying those energy bills! 💅",
            "Your energy consumption said 'I'm that girl' 💋",
            "This is so demure, so mindful, so sustainable ✨",
            "Energy efficiency is your main character moment 🌟"
        ]
        st.info(f"💭 {random.choice(motivation)}")
        
        # Breakdown chart with brainrot
        if ac > 0 or fridge > 0 or wm > 0:
            st.subheader("📈 ENERGY BREAKDOWN (THE VISUAL SLAY)")
            
            breakdown_data = {
                "Source": ["Base (House Vibes)", "Air Conditioners", "Refrigerators", "Washing Machines"],
                "Energy (kWh)": [base_energy[bhk], ac * 3, fridge * 4, wm * 3]
            }
            
            # Filter out zero values for cleaner chart
            filtered_data = {
                "Source": [source for source, energy in zip(breakdown_data["Source"], breakdown_data["Energy (kWh)"]) if energy > 0],
                "Energy (kWh)": [energy for energy in breakdown_data["Energy (kWh)"] if energy > 0]
            }
            
            st.bar_chart(dict(zip(filtered_data["Source"], filtered_data["Energy (kWh)"])))
            
            # Add some brainrot commentary
            st.markdown("**Chart said:** 'I'm not just a chart, I'm THE chart' 💅")
    
    else:
        error_messages = [
            "❌ BESTIE PLEASE FILL OUT ALL THE FIELDS! This is giving incomplete energy 💀",
            "❌ Girl, you need to give us your Name, City, and Area! Don't leave us hanging! 😭",
            "❌ The form is not complete bestie! Fill it out or you're giving Ohio energy 🎭"
        ]
        st.error(random.choice(error_messages))

# Footer with maximum brainrot
st.markdown("---")
st.markdown("### 💀 CREDITS 💀")
st.markdown("Built with ❤️ using Streamlit by your favorite developer (that's me, I'm the main character) 💅")
st.markdown("**Remember:** Stay hydrated, stay sigma, and keep your energy consumption demure 💯")

# Random brainrot fact
facts = [
    "💡 Fun Fact: AC units are the main characters of energy consumption",
    "🔥 Did you know: Fridges run 24/7 because they're always that girl",
    "✨ Energy tip: Washing machines only use energy when they're slaying your clothes",
    "💀 Remember: Every kWh counts in your sustainability era"
]
st.info(random.choice(facts))

# Final brainrot sign-off
st.markdown("### 🎭 THAT'S ALL FOLKS! 🎭")
st.markdown("*No cap, this calculator absolutely sent me* 💀")