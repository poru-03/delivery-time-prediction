import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from pathlib import Path

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="OptiRoute | Dynamic Delivery Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Enterprise-Grade Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Top Navbar / Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(255, 255, 255, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #FFFFFF 0%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #34D399;
        margin-bottom: 0.8rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34D399;
    }

    .author-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #E2E8F0;
        margin-top: 0.5rem;
    }

    /* Cards */
    .custom-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.2rem;
    }

    .card-header-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
    }

    /* Hero Prediction Card */
    .prediction-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 2px solid #3B82F6;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 12px 24px -6px rgba(59, 130, 246, 0.12), 0 4px 10px -2px rgba(0, 0, 0, 0.04);
        text-align: center;
        position: relative;
    }

    .prediction-eyebrow {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 0.2rem;
    }

    .prediction-value {
        font-size: 3.6rem;
        font-weight: 800;
        color: #1D4ED8;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .prediction-unit {
        font-size: 1.4rem;
        font-weight: 600;
        color: #64748B;
        margin-left: 4px;
    }

    .arrival-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #1E40AF;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 6px 16px;
        border-radius: 9999px;
        border: 1px solid #BFDBFE;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Stepper Timeline */
    .timeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        margin: 1.4rem 0 1.2rem 0;
        padding: 0 10px;
    }

    .timeline-line {
        position: absolute;
        top: 18px;
        left: 30px;
        right: 30px;
        height: 3px;
        background: #E2E8F0;
        z-index: 1;
    }

    .timeline-step {
        position: relative;
        z-index: 2;
        text-align: center;
    }

    .step-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 2px solid #CBD5E1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748B;
        margin: 0 auto 6px auto;
    }

    .step-active .step-circle {
        background: #2563EB;
        border-color: #2563EB;
        color: white;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.2);
    }

    .step-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
    }

    .step-desc {
        font-size: 0.72rem;
        color: #94A3B8;
    }

    /* Risk Cards */
    .risk-banner {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    
    .risk-banner-low {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }

    .risk-banner-med {
        background-color: #FEFCE8;
        border: 1px solid #FEF08A;
        color: #854D0E;
    }

    .risk-banner-high {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }

    /* Preset Pill Buttons */
    .preset-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    /* Stat Box */
    .stat-mini-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-mini-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
    }
    .stat-mini-lbl {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL INGESTION & DATA PREPARATION
# ==============================================================================
@st.cache_resource
def load_trained_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "final_model.pkl"
    if not model_path.exists():
        model_path = Path("models/final_model.pkl")
    return joblib.load(model_path)

try:
    model = load_trained_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading trained model pipeline: {e}")

# Brazilian State metadata
state_dict = {
    "SP": "São Paulo (Southeast Hub)",
    "RJ": "Rio de Janeiro (Southeast)",
    "MG": "Minas Gerais (Southeast)",
    "RS": "Rio Grande do Sul (South)",
    "PR": "Paraná (South)",
    "SC": "Santa Catarina (South)",
    "BA": "Bahia (Northeast)",
    "DF": "Distrito Federal (Center)",
    "GO": "Goiás (Center-West)",
    "ES": "Espírito Santo (Southeast)",
    "PE": "Pernambuco (Northeast)",
    "CE": "Ceará (Northeast)",
    "PA": "Pará (North)",
    "MT": "Mato Grosso (Center-West)",
    "MA": "Maranhão (Northeast)",
    "MS": "Mato Grosso do Sul (Center-West)",
    "PB": "Paraíba (Northeast)",
    "PI": "Piauí (Northeast)",
    "RN": "Rio Grande do Norte (Northeast)",
    "AL": "Alagoas (Northeast)",
    "SE": "Sergipe (Northeast)",
    "TO": "Tocantins (North)",
    "RO": "Rondônia (North)",
    "AM": "Amazonas (North)",
    "AC": "Acre (North)",
    "AP": "Amapá (North)",
    "RR": "Roraima (North)"
}

# Category formatting map
category_labels = {
    "health_beauty": "Health & Beauty",
    "bed_bath_table": "Bed, Bath & Table",
    "sports_leisure": "Sports & Leisure",
    "computers_accessories": "Computers & Tech Accessories",
    "furniture_decor": "Furniture & Home Decor",
    "housewares": "Housewares",
    "watches_gifts": "Watches & Premium Gifts",
    "telephony": "Telephony & Smartphones",
    "auto": "Automotive Parts & Accessories",
    "toys": "Toys & Games",
    "cool_stuff": "Cool Stuff & Gadgets",
    "garden_tools": "Garden Tools",
    "perfumery": "Perfumery & Fragrance",
    "baby": "Baby Products",
    "electronics": "Consumer Electronics",
    "stationery": "Stationery & Office Supplies",
    "fashion_bags_accessories": "Fashion Bags & Accessories",
    "office_furniture": "Office Furniture (Bulky)",
    "pet_shop": "Pet Shop Supplies",
    "luggage_accessories": "Luggage & Travel Accessories",
    "consoles_games": "Gaming Consoles & Titles",
    "home_appliances": "Home Appliances",
    "small_appliances": "Small Kitchen Appliances",
    "food": "Food & Gourmet",
    "drinks": "Beverages & Drinks",
    "books_general_interest": "Books (General Interest)",
    "audio": "Audio & Sound Gear",
    "unknown": "Other / Uncategorized"
}

# Inverse category lookup
category_values = {v: k for k, v in category_labels.items()}

# ==============================================================================
# 3. SESSION STATE & QUICK SCENARIO PRESETS
# ==============================================================================
def apply_preset(state_val, same_state_val, cat_val, price_val, freight_val, weight_val, seller_val, day_val):
    st.session_state["p_state"] = state_val
    st.session_state["p_geo"] = same_state_val
    st.session_state["p_cat"] = cat_val
    st.session_state["p_price"] = price_val
    st.session_state["p_freight"] = freight_val
    st.session_state["p_weight"] = weight_val
    st.session_state["p_seller"] = seller_val
    st.session_state["p_day"] = day_val

# Initialize session defaults
if "p_state" not in st.session_state:
    st.session_state["p_state"] = "SP"
if "p_geo" not in st.session_state:
    st.session_state["p_geo"] = "Intra-State (Same State Hub)"
if "p_cat" not in st.session_state:
    st.session_state["p_cat"] = "Health & Beauty"
if "p_price" not in st.session_state:
    st.session_state["p_price"] = 120.0
if "p_freight" not in st.session_state:
    st.session_state["p_freight"] = 16.5
if "p_weight" not in st.session_state:
    st.session_state["p_weight"] = 800
if "p_seller" not in st.session_state:
    st.session_state["p_seller"] = 7.5
if "p_day" not in st.session_state:
    st.session_state["p_day"] = "Monday"

# ==============================================================================
# 4. TOP HERO HEADER & METADATA
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="status-badge">
        <div class="status-dot"></div>
        PRODUCTION MODEL ONLINE • TUNED XGBOOST v3.4.2
    </div>
    <div class="hero-title">OptiRoute | Dynamic Delivery Prediction System</div>
    <div class="hero-subtitle">
        Operational Machine Learning Decision Support System for Last-Mile E-Commerce Logistics. 
        Replaces rigid rule-based delivery estimates with dynamic, checkout-time intelligence accounting for 
        geographical friction, merchant track record, and parcel physical attributes.
    </div>
    <div class="author-chip">
        🎓 <strong>Developer:</strong> M M D H Malporu (IM/2023/015) &nbsp;•&nbsp; 
        Data Science Group 1 &nbsp;•&nbsp; 
        Department of Industrial Management, University of Kelaniya
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. NAVIGATION TABS
# ==============================================================================
tab_calculator, tab_scenarios, tab_metrics, tab_insights = st.tabs([
    "🚀 Live Delivery Estimator",
    "🔄 Scenario Comparison Matrix",
    "📈 Model Evaluation & Scorecard",
    "💡 Business Insights & Methodology"
])

# ==============================================================================
# TAB 1: LIVE DELIVERY ESTIMATOR
# ==============================================================================
with tab_calculator:
    # Preset Selector Bar
    st.markdown('<div class="preset-title">⚡ Quick Scenario Presets (Click to Load)</div>', unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("⚡ São Paulo Local Express", use_container_width=True):
            apply_preset("SP", "Intra-State (Same State Hub)", "Health & Beauty", 89.0, 12.0, 450, 6.0, "Monday")
            st.rerun()
    with col_p2:
        if st.button("🚚 Interstate Tech (Rio)", use_container_width=True):
            apply_preset("RJ", "Cross-State (Interstate Transit)", "Computers & Tech Accessories", 350.0, 28.5, 1800, 11.0, "Tuesday")
            st.rerun()
    with col_p3:
        if st.button("📦 Bulky Furniture (Minas)", use_container_width=True):
            apply_preset("MG", "Cross-State (Interstate Transit)", "Furniture & Home Decor", 480.0, 54.0, 14500, 13.5, "Wednesday")
            st.rerun()
    with col_p4:
        if st.button("⚠️ Remote Northeast (Bahia)", use_container_width=True):
            apply_preset("BA", "Cross-State (Interstate Transit)", "Home Appliances", 260.0, 68.0, 7200, 18.0, "Friday")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2-Column Responsive Workspace
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        # Card 1: Geographic Alignment
        with st.container():
            st.markdown('<div class="card-header-title">📍 1. Geographic Routing & Destination</div>', unsafe_allow_html=True)
            
            geo_col1, geo_col2 = st.columns(2)
            with geo_col1:
                selected_state = st.selectbox(
                    "Customer Destination State",
                    options=list(state_dict.keys()),
                    format_func=lambda x: f"{x} - {state_dict[x]}",
                    index=list(state_dict.keys()).index(st.session_state["p_state"])
                )
            with geo_col2:
                geo_mode = st.radio(
                    "Fulfillment Routing",
                    ["Intra-State (Same State Hub)", "Cross-State (Interstate Transit)"],
                    index=0 if "Same State" in st.session_state["p_geo"] else 1
                )
            
            same_state_flag = 1 if "Same State" in geo_mode else 0
            pct_same_state = 1.0 if same_state_flag == 1 else 0.0

        # Card 2: Package & Economics
        with st.container():
            st.markdown('<div class="card-header-title">🏷️ 2. Product Characteristics & Package Dimensions</div>', unsafe_allow_html=True)
            
            selected_cat_label = st.selectbox(
                "Product Category",
                options=list(category_labels.values()),
                index=list(category_labels.values()).index(st.session_state["p_cat"])
            )
            raw_top_category = category_values[selected_cat_label]

            eco_col1, eco_col2, eco_col3 = st.columns(3)
            with eco_col1:
                price = st.number_input("Item Price (R$)", min_value=5.0, max_value=5000.0, value=float(st.session_state["p_price"]), step=10.0)
            with eco_col2:
                freight = st.number_input("Freight Cost (R$)", min_value=2.0, max_value=500.0, value=float(st.session_state["p_freight"]), step=1.0)
            with eco_col3:
                weight_g = st.number_input("Weight (Grams)", min_value=50, max_value=40000, value=int(st.session_state["p_weight"]), step=100)

            dim_col1, dim_col2, dim_col3 = st.columns(3)
            with dim_col1:
                length_cm = st.slider("Length (cm)", 5, 100, 25)
            with dim_col2:
                height_cm = st.slider("Height (cm)", 5, 100, 15)
            with dim_col3:
                width_cm = st.slider("Width (cm)", 5, 100, 20)
            
            volume_cm3 = float(length_cm * height_cm * width_cm)
            volume_liters = volume_cm3 / 1000.0
            st.caption(f"📏 Calculated Package Volume: **{volume_cm3:,.0f} cm³** ({volume_liters:.1f} Liters)")

        # Card 3: Merchant & Scheduling
        with st.container():
            st.markdown('<div class="card-header-title">⏱️ 3. Merchant Reliability & Timing</div>', unsafe_allow_html=True)
            
            timing_col1, timing_col2 = st.columns(2)
            with timing_col1:
                seller_history = st.slider(
                    "Merchant Historical Latency (Days)",
                    min_value=3.0, max_value=30.0,
                    value=float(st.session_state["p_seller"]),
                    step=0.5,
                    help="Historical average fulfillment latency for this merchant. Strongest operational delay predictor."
                )
                if seller_history < 8.0:
                    st.caption("⚡ **High-Speed Merchant:** Typically dispatches in under 48 hours.")
                elif seller_history <= 14.0:
                    st.caption("⚖️ **Standard Merchant:** Average dispatch within 3-5 days.")
                else:
                    st.caption("🐢 **High-Latency Merchant:** History of delayed carrier handoffs.")

            with timing_col2:
                order_day = st.selectbox(
                    "Order Placement Day",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(st.session_state["p_day"])
                )
                day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
                order_day_idx = day_map[order_day]
                is_weekend = 1 if order_day_idx in [5, 6] else 0

    # Right Column: Live Decision Deck
    with right_col:
        if model_loaded:
            # Construct inference DataFrame
            input_df = pd.DataFrame([{
                "num_items": 1,
                "num_unique_sellers": 1,
                "num_unique_products": 1,
                "total_price": float(price),
                "avg_price": float(price),
                "total_freight_value": float(freight),
                "avg_freight_value": float(freight),
                "avg_product_weight_g": float(weight_g),
                "avg_product_volume_cm3": float(volume_cm3),
                "pct_items_same_state": float(pct_same_state),
                "seller_avg_delivery_days": float(seller_history),
                "same_state_delivery": int(same_state_flag),
                "order_day_of_week": int(order_day_idx),
                "is_weekend_order": int(is_weekend),
                "customer_state": selected_state,
                "top_category": raw_top_category
            }])

            # Generate prediction
            predicted_days = float(model.predict(input_df)[0])
            predicted_days = max(1.0, predicted_days)

            # Date calculation
            today = datetime.date(2026, 9, 28)
            est_arrival_date = today + datetime.timedelta(days=int(round(predicted_days)))
            
            # SLA Window (Model Test MAE is ~4.56d; conservative buffer ±2.5d)
            win_min = max(1, int(round(predicted_days - 2.0)))
            win_max = int(round(predicted_days + 3.0))
            
            # SLA Tier
            if predicted_days < 7.0:
                tier_label = "EXPRESS LOGISTICS"
                tier_color = "#10B981"
            elif predicted_days <= 14.0:
                tier_label = "STANDARD DELIVERY"
                tier_color = "#3B82F6"
            else:
                tier_label = "EXTENDED FREIGHT"
                tier_color = "#EF4444"

            # Render Hero Prediction Card
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-eyebrow">Estimated Delivery Duration</div>
                <div class="prediction-value">{predicted_days:.1f}<span class="prediction-unit">days</span></div>
                <div class="arrival-badge">
                    🗓️ Expected Arrival: <strong>{est_arrival_date.strftime('%A, %b %d, %Y')}</strong>
                </div>
                <div style="display: flex; justify-content: center; gap: 8px; margin-bottom: 0.5rem;">
                    <span style="background: {tier_color}18; color: {tier_color}; border: 1px solid {tier_color}40; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">
                        ● {tier_label}
                    </span>
                    <span style="background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">
                        Promised Window: {win_min} – {win_max} Days
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Visual Stepper Timeline
            dispatch_days = min(3.0, round(seller_history * 0.25, 1))
            transit_days = max(1.0, round(predicted_days - dispatch_days, 1))

            st.markdown(f"""
            <div class="custom-card" style="margin-top: 1rem; padding: 1.2rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #1E293B; margin-bottom: 0.6rem;">
                    📦 Predicted Fulfillment Milestones
                </div>
                <div class="timeline-container">
                    <div class="timeline-line"></div>
                    <div class="timeline-step step-active">
                        <div class="step-circle">1</div>
                        <div class="step-title">Placed</div>
                        <div class="step-desc">Day 0</div>
                    </div>
                    <div class="timeline-step step-active">
                        <div class="step-circle">2</div>
                        <div class="step-title">Dispatched</div>
                        <div class="step-desc">~{dispatch_days}d</div>
                    </div>
                    <div class="timeline-step step-active">
                        <div class="step-circle">3</div>
                        <div class="step-title">In Transit</div>
                        <div class="step-desc">~{transit_days}d</div>
                    </div>
                    <div class="timeline-step step-active">
                        <div class="step-circle">4</div>
                        <div class="step-title">Delivered</div>
                        <div class="step-desc">~{predicted_days:.1f}d</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key Delay Drivers Diagnostic Box
            is_remote_state = selected_state in ["BA", "CE", "MA", "PA", "AM", "AL", "PI", "RN", "PB", "AP", "RR"]
            is_slow_seller = seller_history > 14.0
            is_heavy = weight_g > 5000

            if same_state_flag == 1 and not is_slow_seller:
                st.markdown("""
                <div class="risk-banner risk-banner-low">
                    <div style="font-size: 1.4rem;">✅</div>
                    <div>
                        <strong>Optimal Fulfillment Route:</strong> Intra-state shipping avoids intermediate sorting hubs, enabling rapid last-mile carrier handoff.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif is_remote_state or is_slow_seller:
                st.markdown(f"""
                <div class="risk-banner risk-banner-high">
                    <div style="font-size: 1.4rem;">⚠️</div>
                    <div>
                        <strong>Elevated Delay Risk Detected:</strong><br/>
                        {"• Destination is a remote corridor (" + selected_state + ") with documented linehaul transit friction.<br/>" if is_remote_state else ""}
                        {"• Merchant has an unoptimized dispatch latency history (" + str(seller_history) + " days).<br/>" if is_slow_seller else ""}
                        {"• Package exceeds 5 kg freight threshold, requiring ground cargo handling.<br/>" if is_heavy else ""}
                        <em>Action: Advise operations to communicate the extended {win_min}–{win_max} day SLA to protect customer trust.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-banner risk-banner-med">
                    <div style="font-size: 1.4rem;">ℹ️</div>
                    <div>
                        <strong>Standard Cross-State Transit:</strong> Inter-hub linehaul routing applies. Standard national carrier SLAs will meet customer expectations.
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: SCENARIO COMPARISON MATRIX
# ==============================================================================
with tab_scenarios:
    st.markdown("### 🔄 Multi-Scenario Sensitivity Simulator")
    st.write("Compare how changing key logistics variables shifts delivery latency across 4 realistic business scenarios:")

    scenarios = [
        {"Scenario": "⚡ Intra-State Local (SP)", "State": "SP", "Routing": 1, "Category": "health_beauty", "Weight (g)": 500, "Freight": 12.0, "Seller Hist (d)": 6.0},
        {"Scenario": "🚚 Cross-State Tech (RJ)", "State": "RJ", "Routing": 0, "Category": "computers_accessories", "Weight (g)": 2000, "Freight": 28.0, "Seller Hist (d)": 10.5},
        {"Scenario": "📦 Heavy Cargo (MG)", "State": "MG", "Routing": 0, "Category": "furniture_decor", "Weight (g)": 15000, "Freight": 55.0, "Seller Hist (d)": 14.0},
        {"Scenario": "⚠️ Remote Northeast (BA)", "State": "BA", "Routing": 0, "Category": "office_furniture", "Weight (g)": 18000, "Freight": 72.0, "Seller Hist (d)": 18.0}
    ]

    sim_rows = []
    if model_loaded:
        for sc in scenarios:
            row_df = pd.DataFrame([{
                "num_items": 1, "num_unique_sellers": 1, "num_unique_products": 1,
                "total_price": 150.0, "avg_price": 150.0,
                "total_freight_value": sc["Freight"], "avg_freight_value": sc["Freight"],
                "avg_product_weight_g": sc["Weight (g)"], "avg_product_volume_cm3": 8000.0,
                "pct_items_same_state": float(sc["Routing"]),
                "seller_avg_delivery_days": sc["Seller Hist (d)"],
                "same_state_delivery": sc["Routing"],
                "order_day_of_week": 0, "is_weekend_order": 0,
                "customer_state": sc["State"],
                "top_category": sc["Category"]
            }])
            pred = float(model.predict(row_df)[0])
            sim_rows.append({
                "Scenario": sc["Scenario"],
                "State": sc["State"],
                "Fulfillment": "Same-State" if sc["Routing"] == 1 else "Cross-State",
                "Weight (kg)": f"{sc['Weight (g)']/1000:.1f}",
                "Freight (R$)": f"R$ {sc['Freight']:.1f}",
                "Seller Latency": f"{sc['Seller Hist (d)']} d",
                "Estimated Delivery": f"{pred:.1f} Days",
                "SLA Window": f"{max(1, int(round(pred-2)))} – {int(round(pred+3))} Days"
            })
    
    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True)
    st.info("💡 **Key Takeaway:** Intra-state fulfillment is the single most powerful operational lever to cut customer delivery times by up to 50%.")

# ==============================================================================
# TAB 3: MODEL SCORECARD & EVALUATION
# ==============================================================================
with tab_metrics:
    st.markdown("### 📈 Machine Learning Model Evaluation & Benchmark Scorecard")
    st.write("Rigorous multi-model comparison across 77,168 training orders and 19,292 held-out test orders:")

    # Top metrics display
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="stat-mini-box">
            <div class="stat-mini-val" style="color: #10B981;">4.56 d</div>
            <div class="stat-mini-lbl">TEST MAE (TUNED XGBOOST)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="stat-mini-box">
            <div class="stat-mini-val" style="color: #3B82F6;">0.325</div>
            <div class="stat-mini-lbl">TEST R² SCORE</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="stat-mini-box">
            <div class="stat-mini-val" style="color: #6366F1;">29.1%</div>
            <div class="stat-mini-lbl">ERROR REDUCTION VS BASELINE</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="stat-mini-box">
            <div class="stat-mini-val" style="color: #0F172A;">< 1 ms</div>
            <div class="stat-mini-lbl">INFERENCE LATENCY</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full table
    comp_data = pd.DataFrame([
        {"Model": "Naive Mean Baseline", "Train MAE": "6.412 d", "5-Fold CV MAE": "6.412 ± 0.036", "Test MAE": "6.433 d", "Test RMSE": "9.739 d", "Test R²": "0.000", "Generalization": "Healthy"},
        {"Model": "Ridge Regression", "Train MAE": "5.103 d", "5-Fold CV MAE": "5.111 ± 0.041", "Test MAE": "5.101 d", "Test RMSE": "8.413 d", "Test R²": "0.254", "Generalization": "Healthy"},
        {"Model": "Random Forest (50 trees)", "Train MAE": "4.411 d", "5-Fold CV MAE": "4.767 ± 0.032", "Test MAE": "4.706 d", "Test RMSE": "8.153 d", "Test R²": "0.299", "Generalization": "Healthy"},
        {"Model": "XGBoost (Baseline)", "Train MAE": "4.604 d", "5-Fold CV MAE": "4.804 ± 0.046", "Test MAE": "4.754 d", "Test RMSE": "8.154 d", "Test R²": "0.299", "Generalization": "Healthy"},
        {"Model": "Tuned XGBoost (Selected)", "Train MAE": "4.312 d", "5-Fold CV MAE": "4.617 ± 0.038", "Test MAE": "4.560 d", "Test RMSE": "8.002 d", "Test R²": "0.325", "Generalization": "Optimal"}
    ])
    st.table(comp_data)

    st.markdown(r"""
    #### 🏆 Why Tuned XGBoost Was Selected for Production:
    1. **Predictive Accuracy:** Achieves the lowest absolute error (**4.560 days**) and highest variance explained ($R^2 = 0.325$).
    2. **Proven Stability:** 5-fold cross-validation standard deviation is just $\pm 0.038$ days across data partitions.
    3. **Generalization Gap:** Test MAE vs Train MAE gap is only $+0.248$ days ($+5.7\%$), confirming healthy learning without memorization.
    4. **Sub-Millisecond Inference:** Light enough to embed directly into high-throughput e-commerce checkout flows.
    """)

# ==============================================================================
# TAB 4: BUSINESS INSIGHTS & METHODOLOGY
# ==============================================================================
with tab_insights:
    st.markdown("### 💡 Core Business Insights & Logistics Recommendations")
    
    st.markdown("""
    1. **Geography is the Strongest Determinant:** Intra-state deliveries average **7.95 days**, compared to **15.13 days** for cross-state deliveries. Establishing regional fulfillment centers near major customer hubs (São Paulo and Rio de Janeiro) can halve shipping duration.
    2. **Seller Historical Performance Dictates Delays:** Historical seller delivery latency ($r = +0.27$) is the single largest operational friction driver. Introducing merchant dispatch scorecards will directly boost customer satisfaction.
    3. **Freight Value Acts as a Transit Distance Proxy:** Higher freight charges strongly correlate with cross-country transit ($r = +0.22$).
    4. **Product Categories Exhibit Clear Speed Profiles:** Office Furniture is the slowest category (~20.7 days), while Food and Health & Beauty are the fastest (~9.8 days).
    5. **Weekend Invariance:** Weekend order placement has near-zero correlation with delivery time ($r \approx 0.0039$), confirming that dispatch workflows smoothly absorb weekend volume on Mondays.
    """)

    st.markdown("---")
    st.markdown("""
    #### 🎓 Academic & Team Credentials
    * **Student Name:** M M D H Malporu
    * **Student ID:** IM/2023/015
    * **Course:** Machine Learning Fundamentals & Evaluation (Weeks 07–08)
    * **Group:** Data Science Group 1
    * **Institution:** Department of Industrial Management, Faculty of Science, University of Kelaniya
    * **Dataset:** Olist Brazilian E-Commerce Dataset (99,441 orders)
    """)
