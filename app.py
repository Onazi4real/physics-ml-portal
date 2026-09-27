import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set up clean, responsive wide page layout
st.set_page_config(page_title="Unified Physics ML Portal", layout="wide")

st.title("🚀 Unified Physics Machine Learning Portal")
st.write("Deploying advanced astrophysics classification and material science regression models to the cloud.")

# Initialize the dual-tab workspace architecture
tab1, tab2 = st.tabs(["🌌 ExoPlanetary Signal Classifier", "⚛️ Superconductivity Thermal Simulator"])

# Image URLs for live prediction visualizations
URL_PLANET = "https://unsplash.com"  # Vibrant Earth-like planet
URL_NOISE = "https://unsplash.com"   # Static noise/glitch pattern

# ==========================================
# TAB 1: EXOPLANET SIGNAL CLASSIFICATION
# ==========================================
with tab1:
    st.header("🌌 Kepler Space Telescope Transit Analyzer")
    st.write("Input raw continuous physical telemetry from target stellar coordinates to classify true planet orbits vs system noise.")
    
    try:
        # Load classification binary structures from the repo level
        exo_model = joblib.load("exoplanet_model.pkl")
        exo_scaler = joblib.load("exoplanet_scaler.pkl")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔭 Telescope Telemetry Inputs")
            raw_period = st.number_input("Stellar Orbital Period (Days)", min_value=0.1, max_value=150000.0, value=9.75, key="exo_period")
            raw_depth = st.number_input("Transit Dimming Light Depth (ppm)", min_value=0.0, max_value=2000000.0, value=421.0, key="exo_depth")
            kep_mag = st.number_input("Kepler Host Star Magnitude (Brightness Scale)", min_value=1.0, max_value=25.0, value=14.52, key="exo_mag")
            
        with col2:
            st.subheader("🎯 Live Inference Dashboard")
            
            # 1. Enforce the mathematical log-transformations derived during research phase
            period_log = np.log1p(raw_period)
            depth_log = np.log1p(raw_depth)
            
            # 2. DYNAMIC PHYSICS RECONSTRUCTION:
            # Calculate planet radius natively using the geometric area formula instead of a static value!
            star_radius = 0.96  # Median Kepler star radius in solar units
            computed_prad = star_radius * np.sqrt(raw_depth / 1000000.0) * 109.2  # Convert solar radii to Earth radii
            
            # Calculate a realistic signal-to-noise ratio based on light depth
            computed_snr = np.sqrt(raw_depth) * 1.16 if raw_depth > 0 else 0.0
            
            # Construct the complete 36-feature row cleanly matching your exact training template
            exo_data = {
                'koi_period_err1': [0.0], 'koi_period_err2': [0.0], 'koi_time0bk': [137.2], 
                'koi_time0bk_err1': [0.0], 'koi_time0bk_err2': [0.0], 'koi_impact': [0.537], 
                'koi_impact_err1': [0.0], 'koi_impact_err2': [0.0], 'koi_duration': [3.73], 
                'koi_duration_err1': [0.0], 'koi_duration_err2': [0.0], 'koi_depth_err1': [0.0], 
                'koi_depth_err2': [0.0], 'koi_prad': [computed_prad], 'koi_prad_err1': [0.0], 
                'koi_prad_err2': [0.0], 'koi_teq': [888.0], 'koi_insol': [146.9], 
                'koi_insol_err1': [0.0], 'koi_insol_err2': [0.0], 'koi_model_snr': [computed_snr], 
                'koi_tce_plnt_num': [1.0], 'koi_steff': [5757.0], 'koi_steff_err1': [0.0], 
                'koi_steff_err2': [0.0], 'koi_slogg': [4.43], 'koi_slogg_err1': [0.0], 
                'koi_slogg_err2': [0.0], 'koi_srad': [star_radius], 'koi_srad_err1': [0.0], 
                'koi_srad_err2': [0.0], 'ra': [292.26], 'dec': [43.67], 'koi_kepmag': [kep_mag], 
                'koi_period_log': [period_log], 'koi_depth_log': [depth_log]
            }
            
            X_exo = pd.DataFrame(exo_data)
            
            # Normalize and Predict
            scaled_input = exo_scaler.transform(X_exo)
            prediction = exo_model.predict(scaled_input)
            probability = exo_model.predict_proba(scaled_input)
            
            # Display Prediction Probabilities
            st.metric(label="Calculated Probability of Confirmed Exoplanet", value=f"{probability[0][1]*100:.2f}%")
            
            # 3. DYNAMIC METRIC OUTCOMES & IMAGERY BLOCK
            if prediction == 1 and raw_depth < 30000:  # Logical physics constraint check for extreme eclipsing binaries
                st.success("🟩 PREDICTED STATUS: CONFIRMED EXOPLANET CANDIDATE")
                st.image(URL_PLANET, caption="Stellar System Simulation: True Orbiting Exoplanet Detected.", width=350)
            else:
                st.error("🟥 PREDICTED STATUS: INSTRUMENTAL NOISE / FALSE POSITIVE")
                st.image(URL_NOISE, caption="Telemetry Signal Pattern: Non-Planetary False Alarm or Eclipsing Binary Noise.", width=350)
                
    except Exception as e:
        st.error(f"Initialization Error in Tab 1: {e}")

# ==========================================
# TAB 2: SUPERCONDUCTIVITY THERMAL SIMULATOR
# ==========================================
with tab2:
    st.header("⚛️ Superconductivity Critical Temperature Regressor")
    st.write("Simulate atomic matrices by evaluating combined weighted structural features to predict phase transformation thresholds.")
    
    try:
        # Load regression binary structures from the repo level
        super_model = joblib.load("superconductivity_model.pkl")
        super_scaler = joblib.load("superconductivity_scaler.pkl")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧬 Atomic Lattice Properties")
            num_elements = st.slider("Number of Unique Elements in Matrix", min_value=1, max_value=9, value=4, key="super_elem")
            mean_mass = st.number_input("Mean Atomic Mass (amu)", min_value=1.0, max_value=250.0, value=87.52, key="super_mass")
            mean_radius = st.number_input("Mean Atomic Radius (pm)", min_value=10.0, max_value=300.0, value=141.2, key="super_rad")
            
        with col2:
            st.subheader("🔥 Predicted Thermal Phase Transformation")
            
            # Construct a complete baseline DataFrame row matching the exact 81 structural columns
            super_base = {
                'number_of_elements': [num_elements], 'mean_atomic_mass': [mean_mass], 'wtd_mean_atomic_mass': [73.2],
                'gmean_atomic_mass': [57.7], 'wtd_gmean_atomic_mass': [52.8], 'entropy_atomic_mass': [1.25],
                'wtd_entropy_atomic_mass': [1.19], 'range_atomic_mass': [74.5], 'wtd_range_atomic_mass': [33.7],
                'std_atomic_mass': [31.5], 'wtd_std_atomic_mass': [30.1], 'mean_fie': [737.1], 'wtd_mean_fie': [769.3],
                'gmean_fie': [709.8], 'wtd_gmean_fie': [745.0], 'entropy_fie': [1.26], 'wtd_entropy_fie': [1.14],
                'range_fie': [775.4], 'wtd_range_fie': [461.3], 'std_fie': [324.9], 'wtd_std_fie': [307.9],
                'mean_atomic_radius': [mean_radius], 'wtd_mean_atomic_radius': [130.6], 'gmean_atomic_radius': [114.7],
                'wtd_gmean_atomic_radius': [114.1], 'entropy_atomic_radius': [1.24], 'wtd_entropy_atomic_radius': [1.17],
                'range_atomic_radius': [152.0], 'wtd_range_atomic_radius': [65.9], 'std_atomic_radius': [63.6],
                'wtd_std_atomic_radius': [61.6], 'mean_Density': [3200.4], 'wtd_mean_Density': [4398.9],
                'gmean_Density': [1314.1], 'wtd_gmean_Density': [2337.5], 'entropy_Density': [1.12],
                'wtd_entropy_Density': [1.02], 'range_Density': [5101.4], 'wtd_range_Density': [2134.8],
                'std_Density': [2104.2], 'wtd_std_Density': [2121.3], 'mean_ElectronAffinity': [80.5],
                'wtd_mean_ElectronAffinity': [94.5], 'gmean_ElectronAffinity': [43.6], 'wtd_gmean_ElectronAffinity': [59.7],
                'entropy_ElectronAffinity': [1.09], 'wtd_entropy_ElectronAffinity': [1.01], 'range_ElectronAffinity': [141.4],
                'wtd_range_ElectronAffinity': [61.7], 'std_ElectronAffinity': [58.4], 'wtd_std_ElectronAffinity': [56.3],
                'mean_FusionHeat': [14.3], 'wtd_mean_FusionHeat': [14.1], 'gmean_FusionHeat': [6.0],
                'wtd_gmean_FusionHeat': [7.0], 'entropy_FusionHeat': [1.04], 'wtd_entropy_FusionHeat': [0.93],
                'range_FusionHeat': [26.0], 'wtd_range_FusionHeat': [10.2], 'std_FusionHeat': [10.3],
                'wtd_std_FusionHeat': [9.7], 'mean_ThermalConductivity': [89.3], 'wtd_mean_ThermalConductivity': [105.1],
                'gmean_ThermalConductivity': [8.9], 'wtd_gmean_ThermalConductivity': [14.8], 'entropy_ThermalConductivity': [0.71],
                'wtd_entropy_ThermalConductivity': [0.65], 'range_ThermalConductivity': [227.4], 'wtd_range_ThermalConductivity': [72.3],
                'std_ThermalConductivity': [87.9], 'wtd_std_ThermalConductivity': [84.1], 'mean_Valence': [3.12],
                'wtd_mean_Valence': [2.89], 'gmean_Valence': [2.83], 'wtd_gmean_Valence': [2.68],
                'entropy_Valence': [1.13], 'wtd_entropy_Valence': [1.03], 'range_Valence': [2.0],
                'wtd_range_Valence': [1.08], 'std_Valence': [0.83], 'wtd_std_Valence': [0.81]
            }
            
            X_super = pd.DataFrame(super_base)
            
            # Normalize and Predict
            scaled_super = super_scaler.transform(X_super)
            predicted_tc = super_model.predict(scaled_super)[0]
            
            # Manage non-physical negative boundaries
            if predicted_tc < 0:
                predicted_tc = 0.0
                
            st.subheader("🏆 Estimated Critical Temperature (Tc):")
