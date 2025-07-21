import streamlit as st
import numpy as np
import pickle

# Streamlit UI
st.set_page_config(page_title="LG Customer Segmentation", layout="centered")

# Load pickled model
@st.cache_resource
def load_model():
    with open("lg_cs_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
scaler = model['scaler']
kmeans = model['kmeans_model']
segment_names = model['segment_names']
segment_analysis = model['segment_analysis']
price_recommendations = model['price_recommendations']


st.title("📊 LG Customer Segmentation Predictor")

st.markdown("Enter customer information to predict segment, pricing, and strategy.")

with st.form("customer_form"):
    st.subheader("🧾 Customer Details")

    spending_score = st.slider("Spending Score", min_value=0, max_value=100, value=50)
    membership_years = st.slider("Membership Years", min_value=0.0, max_value=20.0, step=0.1, value=5.0)
    age = st.number_input("Age", min_value=15, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, step=1000, value=50000)
    purchase_frequency = st.number_input("Purchase Frequency (per year)", min_value=1, max_value=50, value=10)
    last_purchase_amount = st.number_input("Last Purchase Amount ($)", min_value=1, max_value=10000, value=200)
    preferred_category = st.selectbox("Preferred Category", [
        'Home Entertainment', 'Home Appliances', 'Mobile & Personal Devices',
        'Business Solutions', 'Automotive & Mobility Technologies'
    ])

    submitted = st.form_submit_button("🔍 Predict Segment")

if submitted:
    # Step 1: Segment Prediction
    input_features = np.array([[spending_score, membership_years]])
    input_scaled = scaler.transform(input_features)
    segment = kmeans.predict(input_scaled)[0]
    segment_name = segment_names[segment]

    st.success(f"🎯 Predicted Segment: **{segment_name}**")

    # Step 2: Pricing Recommendation
    rec = price_recommendations.get(segment_name)
    if rec:
        st.markdown(f"""
        ### 💰 Price Optimization
        - **Current Avg Price**: ${rec['current_price']:.2f}
        - **Optimal Price**: ${rec['optimal_price']:.2f}
        - **Recommended Change**: {rec['price_change']:+.1f}%
        - **Price Elasticity**: {rec['elasticity']}
        """)

    # Step 3: Strategy Recommendation
    if spending_score > 70:
        strategy = "Premium product focus, exclusive offers, VIP treatment"
    elif purchase_frequency < 8:
        strategy = "Re-engagement campaigns, personalized discounts, product trials"
    elif membership_years > 6:
        strategy = "Loyalty rewards, referral programs, early access to new products"
    else:
        strategy = "Value proposition focus, bundle deals, educational content"

    st.markdown(f"""
    ### 📈 Marketing Strategy
    - **Top Category**: {preferred_category}
    - **Strategy**: {strategy}
    """)

    # Step 4: Summary
    st.markdown("---")
    st.markdown("✅ Analysis complete. Use this insight for campaign personalization or pricing strategy.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | LG Customer Intelligence Suite")