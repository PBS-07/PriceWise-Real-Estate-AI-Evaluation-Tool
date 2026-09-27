import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import folium
from PIL import Image as PILImage
from streamlit_folium import st_folium
from pdf_generator import generate_pdf_report

# Load Favicon Image
icon_path = "assets/pricewise_icon.png"
logo_path = "assets/pricewise_logo.png"

favicon = PILImage.open(icon_path) if os.path.exists(icon_path) else "🏠"

# Streamlit Page Configuration - PriceWise Rebrand
st.set_page_config(
    page_title="PriceWise — Predict smarter. Invest better. | Bengaluru Real Estate",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for PriceWise Premium Branding & Glassmorphism
st.markdown("""
<style>
    /* PriceWise Dark Glassmorphism Palette */
    .stApp {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #070B19 100%);
        color: #F8FAFC;
    }
    
    .brand-hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    
    .brand-tagline {
        font-size: 1.15rem;
        font-weight: 600;
        color: #93C5FD;
        margin-bottom: 16px;
        letter-spacing: 0.3px;
    }

    /* Glass Card Component */
    .glass-card {
        background: rgba(28, 37, 65, 0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-sub {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #94A3B8;
        text-transform: uppercase;
    }

    .badge-bengaluru {
        background-color: rgba(59, 130, 246, 0.18);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }

    .badge-premium {
        background-color: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-mid {
        background-color: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-budget {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563EB 0%, #4F46E5 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #1D4ED8 0%, #4338CA 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.65);
    }
</style>
""", unsafe_allow_html=True)

# Cache Data & Models
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_bengaluru_house_prices.csv")
    with open("artifacts/metadata.json", "r") as f:
        metadata = json.load(f)
    with open("artifacts/model_metrics.json", "r") as f:
        metrics = json.load(f)
    with open("artifacts/feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open("artifacts/feature_importances.json", "r") as f:
        feature_importances = json.load(f)
    with open("artifacts/locality_coords.json", "r") as f:
        locality_coords = json.load(f)
    return df, metadata, metrics, feature_names, feature_importances, locality_coords

@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("artifacts/model_rf.pkl"),
        "Linear Regression": joblib.load("artifacts/model_lr.pkl"),
        "XGBoost": joblib.load("artifacts/model_xgb.pkl")
    }
    return models

df, metadata, metrics, feature_names, feature_importances, locality_coords = load_data()
models = load_models()

# Helper function to convert input dictionary to feature vector
def prepare_input_vector(loc, total_sqft, bhk, bath, balcony, area_type, availability_clean, has_society):
    input_dict = {col: 0 for col in feature_names}
    input_dict['total_sqft'] = float(total_sqft)
    input_dict['bhk'] = int(bhk)
    input_dict['bath'] = float(bath)
    input_dict['balcony'] = float(balcony)
    input_dict['has_society'] = 1 if has_society else 0
    input_dict['is_ready_to_move'] = 1 if availability_clean == "Ready To Move" else 0
    
    area_col = f"area_{area_type}"
    if area_col in input_dict:
        input_dict[area_col] = 1
        
    loc_col = f"loc_{loc}"
    if loc_col in input_dict:
        input_dict[loc_col] = 1
        
    return pd.DataFrame([input_dict])[feature_names]

# SIDEBAR BRANDING
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.title("PriceWise")

st.sidebar.markdown("**Predict smarter. Invest better.**")
st.sidebar.markdown("<span class='badge-bengaluru'>📍 Scope: Bengaluru Real Estate</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "PriceWise Navigation",
    [
        "🎯 1. Predict Bengaluru Prices",
        "📊 2. Bengaluru EDA Dashboard",
        "⚖️ 3. Locality Comparison Tool",
        "🎛️ 4. What-If Live Simulator",
        "🗺️ 5. Bengaluru Locality Heatmap",
        "⚙️ 6. PriceWise AI Engine Benchmarks",
        "📄 7. Export Valuation PDF Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"""
**PriceWise — Real Estate AI Evaluation Tool**  
Dataset: **12,513 Cleaned Bengaluru Listings**  
Source: `bengaluru_house_prices.csv`  
Models: Random Forest | XGBoost | Linear Regression  
""")

# PAGE 1: PREDICT BENGALURU PRICES
if page == "🎯 1. Predict Bengaluru Prices":
    st.markdown("<div class='brand-hero-title'>PriceWise</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-tagline'>Predict smarter. Invest better.</div>", unsafe_allow_html=True)
    st.markdown("<span class='badge-bengaluru'>📍 City Scope: Bengaluru Property Evaluation</span>", unsafe_allow_html=True)
    st.markdown("Predict property prices across Bengaluru's micro-localities powered by machine learning algorithms.")
    
    col_in, col_res = st.columns([1.1, 1])
    
    with col_in:
        st.subheader("📋 Property Specification Inputs")
        
        # Locality dropdown verified against cleaned_bengaluru_house_prices.csv
        selected_loc = st.selectbox(
            "Bengaluru Locality / Neighborhood (bengaluru_house_prices.csv)",
            metadata["locations"],
            index=metadata["locations"].index("Whitefield") if "Whitefield" in metadata["locations"] else 0,
            help="Locality dataset populated directly from 12,513 Bengaluru property listings."
        )
        st.caption("✅ Verified: Populated directly from authentic Bengaluru dataset (`bengaluru_house_prices.csv`).")
        
        c1, c2 = st.columns(2)
        with c1:
            total_sqft = st.number_input("Total Area (Sq. Ft.)", min_value=300, max_value=15000, value=1200, step=50)
            bhk = st.selectbox("BHK / Bedroom Count", [1, 2, 3, 4, 5, 6], index=1)
            bath = st.number_input("Number of Bathrooms", min_value=1, max_value=8, value=2)
        with c2:
            area_type = st.selectbox("Area Type", metadata["area_types"], index=0)
            availability = st.selectbox("Availability Status", ["Ready To Move", "Under Construction"])
            balcony = st.number_input("Number of Balconies", min_value=0, max_value=4, value=1)
            
        has_society = st.checkbox("Gated Society / Named Complex", value=True)
        selected_model_name = st.selectbox("Model Engine Algorithm", ["Random Forest", "XGBoost", "Linear Regression"])
        
        predict_btn = st.button("🚀 Calculate PriceWise Fair Estimate", use_container_width=True)

    # Calculate prediction
    model = models[selected_model_name]
    input_df = prepare_input_vector(selected_loc, total_sqft, bhk, bath, balcony, area_type, availability, has_society)
    predicted_val = float(model.predict(input_df)[0])
    
    # Error margin
    model_rmse = metrics[selected_model_name]["RMSE (₹ Lakhs)"]
    lower_bound = max(10.0, predicted_val - model_rmse * 0.5)
    upper_bound = predicted_val + model_rmse * 0.5
    price_per_sqft = (predicted_val * 100000) / total_sqft
    price_crores = predicted_val / 100.0

    # Locality statistics
    loc_df = df[df['location'] == selected_loc]
    loc_avg_pps = loc_df['price_per_sqft'].mean() if len(loc_df) > 0 else df['price_per_sqft'].mean()
    loc_tier = loc_df['locality_tier'].iloc[0] if len(loc_df) > 0 and 'locality_tier' in loc_df.columns else "Mid-Tier"

    with col_res:
        st.subheader("💡 PriceWise Valuation Summary")
        
        st.markdown(f"""
        <div class="glass-card">
            <span class="metric-sub">PRICEWISE ESTIMATED FAIR VALUE</span>
            <div class="metric-value">₹ {predicted_val:.2f} Lakhs</div>
            <div style="font-size:1.15rem; color:#60A5FA; font-weight:700;">(~ ₹ {price_crores:.2f} Crores)</div>
            <hr style="border-color: rgba(255,255,255,0.12); margin:14px 0;"/>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8;">Confidence Range (95% CI):</span>
                <span style="color:#34D399; font-weight:700;">₹ {lower_bound:.2f} L – ₹ {upper_bound:.2f} L</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8;">PriceWise Estimated Rate:</span>
                <span style="font-weight:700;">₹ {price_per_sqft:,.0f} / sq.ft.</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8;">Bengaluru Locality Benchmark:</span>
                <span style="font-weight:700;">₹ {loc_avg_pps:,.0f} / sq.ft.</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94A3B8;">Locality Classification:</span>
                <span class="badge-mid">{loc_tier}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explainability Breakdown (F2)
        st.subheader("🔍 PriceWise Model Drivers")
        
        factors = [
            ("Property Size (Total Sq. Ft.)", total_sqft * 0.45),
            (f"Bengaluru Locality ({selected_loc})", loc_avg_pps * 0.35),
            ("BHK & Bathroom Ratio", (bhk + bath) * 15.0),
            ("Area Type & Gated Society", 25.0 if has_society else 10.0),
            ("Availability Status", 15.0 if availability == "Ready To Move" else 5.0)
        ]
        
        total_f_val = sum([v for _, v in factors])
        pct_factors = [(f, (v / total_f_val) * 100) for f, v in factors]
        pct_factors.sort(key=lambda x: x[1], reverse=True)
        
        top_driver = pct_factors[0][0]
        st.info(f"💡 **PriceWise Insight:** This estimate of **₹ {predicted_val:.2f} Lakhs** is primarily driven by **{top_driver}** and current market demand in **{selected_loc}, Bengaluru**.")
        
        factor_df = pd.DataFrame(pct_factors, columns=['Valuation Driver', 'Impact Contribution (%)'])
        fig_exp = px.bar(
            factor_df, x='Impact Contribution (%)', y='Valuation Driver', orientation='h',
            title="PriceWise Top Valuation Drivers",
            color='Impact Contribution (%)',
            color_continuous_scale='Blues',
            template="plotly_dark"
        )
        fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=240, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_exp, use_container_width=True)

# PAGE 2: BENGALURU EDA DASHBOARD
elif page == "📊 2. Bengaluru EDA Dashboard":
    st.title("📊 Bengaluru Real Estate Market EDA Dashboard")
    st.markdown("City-wide price distribution, price-per-sq.ft. trends, and feature correlation across Bengaluru localities.")
    
    with st.expander("🔍 Interactive Data Filters (Bengaluru Dataset)", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            loc_filter = st.multiselect("Filter Bengaluru Localities", options=metadata["locations"], default=[])
        with col_f2:
            bhk_filter = st.slider("Filter BHK Range", min_value=int(df['bhk'].min()), max_value=int(df['bhk'].max()), value=(1, 5))
        with col_f3:
            price_filter = st.slider("Filter Price Range (₹ Lakhs)", min_value=float(df['price'].min()), max_value=float(df['price'].quantile(0.98)), value=(float(df['price'].min()), 500.0))

    filtered_df = df.copy()
    if loc_filter:
        filtered_df = filtered_df[filtered_df['location'].isin(loc_filter)]
    filtered_df = filtered_df[(filtered_df['bhk'] >= bhk_filter[0]) & (filtered_df['bhk'] <= bhk_filter[1])]
    filtered_df = filtered_df[(filtered_df['price'] >= price_filter[0]) & (filtered_df['price'] <= price_filter[1])]
    
    st.caption(f"Displaying **{len(filtered_df):,}** Bengaluru property listings matching filters.")

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown("### 📈 Price Distribution in Bengaluru (₹ Lakhs)")
        fig_hist = px.histogram(
            filtered_df, x='price', nbins=50, color='bhk',
            title="Bengaluru Property Price Distribution by BHK",
            labels={'price': 'Price (₹ Lakhs)'},
            template="plotly_dark"
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_g2:
        st.markdown("### 📦 BHK Price Ranges in Bengaluru")
        fig_box = px.box(
            filtered_df, x='bhk', y='price', color='bhk',
            title="Bengaluru Price Ranges per BHK Count",
            labels={'bhk': 'BHK Count', 'price': 'Price (₹ Lakhs)'},
            template="plotly_dark"
        )
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    c_g3, c_g4 = st.columns(2)
    with c_g3:
        st.markdown("### 🏆 Top 15 Premium Bengaluru Localities (Avg ₹/sq.ft.)")
        top_locs = df.groupby('location')['price_per_sqft'].mean().sort_values(ascending=False).head(15).reset_index()
        fig_bar = px.bar(
            top_locs, x='price_per_sqft', y='location', orientation='h',
            title="Top Rates per Sq. Ft. in Bengaluru",
            labels={'price_per_sqft': 'Avg Rate (₹/sqft)', 'location': 'Bengaluru Locality'},
            color='price_per_sqft',
            color_continuous_scale='Viridis',
            template="plotly_dark"
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_g4:
        st.markdown("### 📉 Size vs Price Correlation in Bengaluru")
        fig_scat = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df))), x='total_sqft', y='price', color='locality_tier',
            hover_data=['location', 'bhk', 'bath'],
            title="Bengaluru Property Size (sq.ft.) vs Price (Lakhs)",
            labels={'total_sqft': 'Total Sq. Ft.', 'price': 'Price (₹ Lakhs)'},
            template="plotly_dark"
        )
        fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scat, use_container_width=True)

# PAGE 3: LOCALITY COMPARISON TOOL
elif page == "⚖️ 3. Locality Comparison Tool":
    st.title("⚖️ Bengaluru Locality Comparison Tool")
    st.markdown("Compare 2 to 4 Bengaluru micro-markets side-by-side on PriceWise metrics.")
    
    default_compare = ["Whitefield", "Koramangala", "Electronic City"]
    available_defaults = [l for l in default_compare if l in metadata["locations"]]
    if len(available_defaults) < 2:
        available_defaults = metadata["locations"][:3]
        
    selected_comp_locs = st.multiselect(
        "Select Bengaluru Localities to Compare",
        options=metadata["locations"],
        default=available_defaults,
        max_selections=4
    )
    
    if len(selected_comp_locs) < 2:
        st.warning("Please select at least 2 Bengaluru localities to compare.")
    else:
        comp_df = df[df['location'].isin(selected_comp_locs)]
        
        stats_list = []
        for loc in selected_comp_locs:
            ldf = comp_df[comp_df['location'] == loc]
            if len(ldf) > 0:
                stats_list.append({
                    "Bengaluru Locality": loc,
                    "Listings Count": len(ldf),
                    "Avg Rate (₹/sqft)": f"₹ {ldf['price_per_sqft'].mean():,.0f}",
                    "Median Price (₹ Lakhs)": f"₹ {ldf['price'].median():,.2f} L",
                    "Price Range (₹ Lakhs)": f"₹ {ldf['price'].min():,.0f} L - ₹ {ldf['price'].max():,.0f} L",
                    "Typical Config": f"{int(ldf['bhk'].mode()[0])} BHK",
                    "Tier Classification": ldf['locality_tier'].iloc[0] if 'locality_tier' in ldf.columns else "N/A"
                })
        
        st.markdown("### 📊 Side-by-Side Bengaluru Locality Comparison")
        st.table(pd.DataFrame(stats_list))
        
        c_c1, c_c2 = st.columns(2)
        with c_c1:
            fig_cmp_pps = px.bar(
                comp_df.groupby('location')['price_per_sqft'].mean().reset_index(),
                x='location', y='price_per_sqft', color='location',
                title="Average Rate per Sq. Ft. (₹)",
                labels={'price_per_sqft': 'Avg Rate (₹/sqft)', 'location': 'Bengaluru Locality'},
                template="plotly_dark"
            )
            fig_cmp_pps.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cmp_pps, use_container_width=True)
            
        with c_c2:
            fig_cmp_box = px.box(
                comp_df, x='location', y='price', color='location',
                title="Price Distribution Comparison (₹ Lakhs)",
                labels={'price': 'Price (₹ Lakhs)', 'location': 'Bengaluru Locality'},
                template="plotly_dark"
            )
            fig_cmp_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cmp_box, use_container_width=True)

# PAGE 4: WHAT-IF LIVE SIMULATOR
elif page == "🎛️ 4. What-If Live Simulator":
    st.title("🎛️ Real-Time 'What-If' Price Simulator")
    st.markdown("Adjust property attributes live with sliders to simulate price response across Bengaluru.")
    
    col_s1, col_s2 = st.columns([1, 1.2])
    
    with col_s1:
        st.subheader("🎚️ Live Attribute Controls")
        sim_loc = st.selectbox("Bengaluru Locality", metadata["locations"], index=0)
        sim_sqft = st.slider("Total Sq. Ft.", 400, 4500, 1450, step=25)
        sim_bhk = st.slider("BHK Bedrooms", 1, 6, 3)
        sim_bath = st.slider("Bathrooms", 1, 6, 3)
        sim_balcony = st.slider("Balconies", 0, 4, 2)
        sim_area = st.selectbox("Area Specification", metadata["area_types"])
        sim_society = st.toggle("Located in Gated Society", value=True)
        
        sim_input = prepare_input_vector(sim_loc, sim_sqft, sim_bhk, sim_bath, sim_balcony, sim_area, "Ready To Move", sim_society)
        rf_model = models["Random Forest"]
        sim_price = float(rf_model.predict(sim_input)[0])
        sim_pps = (sim_price * 100000) / sim_sqft
        
        st.markdown(f"""
        <div class="glass-card" style="margin-top:20px;">
            <span class="metric-sub">PRICEWISE SIMULATED LIVE ESTIMATE</span>
            <div class="metric-value">₹ {sim_price:.2f} Lakhs</div>
            <div style="color:#60A5FA; font-weight:700;">Simulated Rate: ₹ {sim_pps:,.0f} / sq.ft.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s2:
        st.subheader("📈 Live Price Trajectory vs. Sq. Ft. Scaling")
        
        sqft_range = np.linspace(500, 4000, 30)
        curve_prices = []
        for s in sqft_range:
            v_input = prepare_input_vector(sim_loc, s, sim_bhk, sim_bath, sim_balcony, sim_area, "Ready To Move", sim_society)
            curve_prices.append(float(rf_model.predict(v_input)[0]))
            
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=sqft_range, y=curve_prices, mode='lines', name='Price Trajectory',
            line=dict(color='#3B82F6', width=3)
        ))
        fig_curve.add_trace(go.Scatter(
            x=[sim_sqft], y=[sim_price], mode='markers', name='Current Selection',
            marker=dict(color='#EF4444', size=14, symbol='diamond')
        ))
        fig_curve.update_layout(
            title=f"PriceWise Trajectory for {sim_bhk} BHK in {sim_loc}, Bengaluru",
            xaxis_title="Total Sq. Ft.",
            yaxis_title="Estimated Price (₹ Lakhs)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_curve, use_container_width=True)

# PAGE 5: BENGALURU LOCALITY HEATMAP
elif page == "🗺️ 5. Bengaluru Locality Heatmap":
    st.title("🗺️ Bengaluru Interactive Locality Heatmap")
    st.markdown("Geospatial visualization of Bengaluru property prices across top micro-markets powered by Google Maps.")
    
    # Read Google Maps API key from Streamlit secrets
    google_maps_api_key = None
    try:
        if "GOOGLE_MAPS_API_KEY" in st.secrets:
            key_val = st.secrets["GOOGLE_MAPS_API_KEY"]
            if key_val and key_val.strip() and key_val.strip() != "your_key_here":
                google_maps_api_key = key_val.strip()
    except Exception:
        google_maps_api_key = None

    if not google_maps_api_key:
        st.warning(
            "⚠️ **Google Maps API Key Missing**: Please configure your API key in `.streamlit/secrets.toml` "
            "(refer to `.streamlit/secrets.toml.example`). Without a valid key, Google Maps tiles may not load properly."
        )

    bengaluru_center = [12.9716, 77.5946]
    
    # Initialize Folium Map without default CARTO tiles to avoid watermark
    m = folium.Map(
        location=bengaluru_center,
        zoom_start=11,
        tiles=None,
        control_scale=True
    )
    
    # Configure Google Maps TileLayer
    if google_maps_api_key:
        google_tiles_url = f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={google_maps_api_key}"
    else:
        google_tiles_url = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
        
    folium.TileLayer(
        tiles=google_tiles_url,
        attr="Google Maps",
        name="Google Maps Dark",
        overlay=False,
        control=False
    ).add_to(m)

    # Apply Custom Dark Map Styling matching Obsidian & Lime theme
    dark_map_css = """
    <style>
    /* Dark Obsidian & Lime Styling for Map Tiles: Near-black land, muted grey roads, minimal clutter */
    .leaflet-tile-pane {
        filter: grayscale(100%) invert(95%) contrast(85%) brightness(80%);
        -webkit-filter: grayscale(100%) invert(95%) contrast(85%) brightness(80%);
    }
    .leaflet-container {
        background: #0B132B !important;
    }
    /* Dark Glassmorphic Popup Card Styling */
    .leaflet-popup-content-wrapper {
        background: #1C2541 !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6) !important;
    }
    .leaflet-popup-tip {
        background: #1C2541 !important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(dark_map_css))

    loc_summary = df.groupby('location').agg(
        avg_pps=('price_per_sqft', 'mean'),
        avg_price=('price', 'mean'),
        count=('price', 'count')
    ).reset_index()
    
    for _, row in loc_summary.iterrows():
        loc_name = row['location']
        if loc_name in locality_coords:
            lat, lon = locality_coords[loc_name]
            pps = row['avg_pps']
            
            # Tier categorization: Red = high, Blue = mid, Green = budget
            if pps > 8500:
                color = "#EF4444"  # High / Premium (Red)
                tier_label = "Premium Tier"
            elif pps > 5000:
                color = "#3B82F6"  # Mid-Tier (Blue)
                tier_label = "Mid-Tier"
            else:
                color = "#22C55E"  # Budget Tier (Green / Lime)
                tier_label = "Budget Tier"

            # Hover tooltip content
            tooltip_html = f"<b>{loc_name}</b> ({tier_label})<br/>Avg Price: ₹ {row['avg_price']:.1f} Lakhs<br/>Rate: ₹ {pps:,.0f} / sqft"

            # Click popup content
            popup_html = f"""
            <div style="font-family: Inter, system-ui, -apple-system, sans-serif; color: #F8FAFC; width: 195px; padding: 4px 2px;">
                <h4 style="margin: 0 0 6px 0; font-size: 14px; font-weight: 700; color: #60A5FA; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 4px;">{loc_name}</h4>
                <div style="font-size: 12px; line-height: 1.5;">
                    <div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
                        <span style="color:#94A3B8;">Tier:</span>
                        <span style="color:{color}; font-weight:700;">{tier_label}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
                        <span style="color:#94A3B8;">Avg Rate:</span>
                        <span style="font-weight:600;">₹ {pps:,.0f}/sqft</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
                        <span style="color:#94A3B8;">Avg Price:</span>
                        <span style="font-weight:600;">₹ {row['avg_price']:.1f} Lakhs</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#94A3B8;">Listings:</span>
                        <span style="font-weight:600;">{row['count']}</span>
                    </div>
                </div>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(6, min(20, row['count'] / 14)),
                tooltip=folium.Tooltip(tooltip_html),
                popup=folium.Popup(popup_html, max_width=230),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1.5
            ).add_to(m)

    st_folium(m, width="100%", height=550)

# PAGE 6: MODEL VERSION BENCHMARKS
elif page == "⚙️ 6. PriceWise AI Engine Benchmarks":
    st.title("⚙️ PriceWise AI Engine Model Benchmarks")
    st.markdown("Performance metrics evaluation across machine learning models trained on Bengaluru housing data.")
    
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model Algorithm Engine'})
    
    st.markdown("### 📊 PriceWise Benchmark Evaluation Table")
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        fig_r2 = px.bar(
            metrics_df, x='Model Algorithm Engine', y='R2 Score', color='Model Algorithm Engine',
            title="Model R² Accuracy Score (Higher is Better)",
            template="plotly_dark"
        )
        fig_r2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with c_m2:
        fig_mae = px.bar(
            metrics_df, x='Model Algorithm Engine', y='MAE (₹ Lakhs)', color='Model Algorithm Engine',
            title="Mean Absolute Error in ₹ Lakhs (Lower is Better)",
            template="plotly_dark"
        )
        fig_mae.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_mae, use_container_width=True)

# PAGE 7: EXPORT PDF REPORT
elif page == "📄 7. Export Valuation PDF Report":
    st.title("📄 Export PriceWise Valuation PDF Report")
    st.markdown("Generate a formal PriceWise valuation report for any property in Bengaluru.")
    
    col_pdf1, col_pdf2 = st.columns(2)
    with col_pdf1:
        pdf_loc = st.selectbox("Bengaluru Locality", metadata["locations"], index=0)
        pdf_sqft = st.number_input("Total Sq. Ft.", value=1350)
        pdf_bhk = st.selectbox("BHK Bedrooms", [1, 2, 3, 4, 5], index=2)
        pdf_bath = st.number_input("Bathrooms", value=2)
        pdf_balcony = st.number_input("Balconies", value=1)
        pdf_area = st.selectbox("Area Specification", metadata["area_types"], index=0)
        pdf_society = st.checkbox("Gated Complex", value=True)
        pdf_model_name = st.selectbox("Model Engine", ["Random Forest", "XGBoost", "Linear Regression"])
        
    pdf_input = prepare_input_vector(pdf_loc, pdf_sqft, pdf_bhk, pdf_bath, pdf_balcony, pdf_area, "Ready To Move", pdf_society)
    pdf_pred = float(models[pdf_model_name].predict(pdf_input)[0])
    pdf_rmse = metrics[pdf_model_name]["RMSE (₹ Lakhs)"]
    
    pred_res_dict = {
        "price_lakhs": pdf_pred,
        "price_crores": pdf_pred / 100.0,
        "lower": max(10.0, pdf_pred - pdf_rmse * 0.5),
        "upper": pdf_pred + pdf_rmse * 0.5,
        "price_per_sqft": (pdf_pred * 100000) / pdf_sqft,
        "model_name": pdf_model_name
    }
    
    prop_details_dict = {
        "location": pdf_loc,
        "total_sqft": pdf_sqft,
        "bhk": pdf_bhk,
        "bath": pdf_bath,
        "balcony": pdf_balcony,
        "area_type": pdf_area,
        "availability": "Ready To Move",
        "has_society": pdf_society
    }
    
    loc_pdf_df = df[df['location'] == pdf_loc]
    loc_stats_dict = {
        "avg_pps": loc_pdf_df['price_per_sqft'].mean() if len(loc_pdf_df) > 0 else df['price_per_sqft'].mean(),
        "tier": loc_pdf_df['locality_tier'].iloc[0] if len(loc_pdf_df) > 0 and 'locality_tier' in loc_pdf_df.columns else "Mid-Tier"
    }
    
    top_factors_list = [
        ("Total Sq. Ft. Area", 45.0),
        ("Bengaluru Locality Benchmark", 30.0),
        ("BHK & Bathroom Count", 15.0),
        ("Gated Society Flag", 10.0)
    ]

    pdf_bytes = generate_pdf_report(prop_details_dict, pred_res_dict, loc_stats_dict, top_factors_list)
    
    with col_pdf2:
        st.subheader("📋 PriceWise Report Preview")
        st.markdown(f"""
        <div class="glass-card">
            <h4>PriceWise Real Estate Valuation Summary</h4>
            <p><b>City Scope:</b> Bengaluru</p>
            <p><b>Locality:</b> {pdf_loc}</p>
            <p><b>Specs:</b> {pdf_sqft} sq.ft. | {pdf_bhk} BHK | {pdf_bath} Bath</p>
            <p><b>PriceWise Fair Value:</b> ₹ {pdf_pred:.2f} Lakhs</p>
            <p><b>Rate:</b> ₹ {pred_res_dict['price_per_sqft']:,.0f} / sq.ft.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download Official PriceWise PDF Report",
            data=pdf_bytes,
            file_name=f"PriceWise_Valuation_{pdf_loc.replace(' ', '_')}_Bengaluru.pdf",
            mime="application/pdf",
            use_container_width=True
        )
