import streamlit as st

# Set page title
st.title("⚡ Electricity Bill Calculator")

# Create input form
with st.form("bill_calculator"):
    # Personal Information
    st.subheader("Personal Information")
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=25)
    
    # Location Information
    st.subheader("Location Information")
    area = st.text_input("Enter your area:")
    city = st.text_input("Enter your city:")
    
    # House Information
    st.subheader("House Information")
    house_type = st.selectbox("Do you live in a Flat or a Tenement?", 
                             ["Flat", "Tenement"])
    rooms = st.number_input("How many bedrooms do you have?", 
                           min_value=1, max_value=10, value=2)
    ac = st.number_input("How many AC units do you have?", 
                        min_value=0, max_value=10, value=1)
    fridge = st.checkbox("Do you have a fridge?", value=True)
    
    # Submit button
    submitted = st.form_submit_button("Calculate Bill")
    
    # Calculate and display bill when form is submitted
    if submitted:
        # Bill calculation logic
        bill = (rooms + 1) * 1.2 + ac * 3
        if fridge:
            bill += 4
        
        # Display results
        st.success(f"Hello {name}! Your electricity bill has been calculated.")
        
        # Display bill breakdown
        st.subheader("Bill Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {name}")
            st.write(f"**Age:** {age}")
            st.write(f"**Location:** {area}, {city}")
            st.write(f"**House Type:** {house_type}")
        
        with col2:
            st.write(f"**Bedrooms:** {rooms}")
            st.write(f"**AC Units:** {ac}")
            st.write(f"**Fridge:** {'Yes' if fridge else 'No'}")
        
        # Bill calculation breakdown
        st.subheader("Cost Breakdown")
        base_cost = (rooms + 1) * 1.2
        ac_cost = ac * 3
        fridge_cost = 4 if fridge else 0
        
        st.write(f"**Base Cost (Rooms + 1) × 1.2:** ${base_cost:.2f}")
        st.write(f"**AC Cost ({ac} units × 3):** ${ac_cost:.2f}")
        st.write(f"**Fridge Cost:** ${fridge_cost:.2f}")
        
        # Total bill with emphasis
        st.markdown("---")
        st.markdown(f"## **Total Bill: ${bill:.2f}**")

# Add some styling and info
st.markdown("---")
st.info("💡 **How the bill is calculated:**\n"
        "- Base cost: (Number of bedrooms + 1) × $1.20\n"
        "- AC cost: Number of AC units × $3.00\n"
        "- Fridge cost: $4.00 (if you have one)")