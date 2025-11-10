import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pack SGA %Pack Predictor", layout="wide", page_icon="📦")

# Load model
@st.cache_resource
def load_model():
    with open('pack_model.pkl', 'rb') as f:
        return pickle.load(f)

model_package = load_model()
model = model_package['model']
scaler = model_package['scaler']
label_encoders = model_package['label_encoders']
feature_names = model_package['feature_names']
categorical_cols = model_package['categorical_cols']
best_model_name = model_package['best_model_name']
metrics = model_package['metrics']

# Sidebar
st.sidebar.title("📦 Pack SGA Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", 
                       ["🔮 Single Prediction", 
                        "📊 Batch Prediction", 
                        "📈 Model Analytics",
                        "ℹ️ About"])

# ============================================================================
# PAGE 1: Single Prediction
# ============================================================================
if page == "🔮 Single Prediction":
    st.title("🔮 Single %Pack Prediction")
    st.markdown("Enter production parameters to predict packing efficiency")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Date & Shift")
        year = st.number_input("Year", min_value=2024, max_value=2030, value=2025)
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day", 1, 31, 15)
        day_of_week = st.selectbox("Day of Week", list(range(7)), 
                                   format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        shift = st.selectbox("Shift", list(label_encoders['Shift'].classes_))
    
    with col2:
        st.subheader("Equipment")
        furnace = st.number_input("Furnace", min_value=1, max_value=5, value=2)
        line = st.number_input("Line", min_value=1, max_value=10, value=1)
    
    with col3:
        st.subheader("Product Info")
        customer = st.selectbox("Customer", list(label_encoders['Customer'].classes_))
        product_type = st.selectbox("Type", list(label_encoders['Type'].classes_))
        description = st.selectbox("Description", list(label_encoders['Description'].classes_))
        total_fg = st.number_input("Total FG (Bottle)", min_value=0, max_value=500000, value=200000)
    
    if st.button("🎯 Predict %Pack", type="primary", use_container_width=True):
        try:
            # Prepare input
            input_data = pd.DataFrame({
                'Shift': [shift],
                'Furnace': [furnace],
                'Line': [line],
                'Customer': [customer],
                'Type': [product_type],
                'Description': [description],
                'Total FG (Bottle)': [total_fg],
                'Year': [year],
                'Month': [month],
                'Day': [day],
                'DayOfWeek': [day_of_week],
                'Quarter': [quarter]
            })
            
            # Encode categorical
            for col in categorical_cols:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
            
            # Ensure correct order
            input_data = input_data[feature_names]
            
            # Scale
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted %Pack", f"{prediction:.4f}", f"{prediction*100:.2f}%")
            col2.metric("Model Used", best_model_name)
            col3.metric("Model R² Score", f"{metrics['r2_test']:.4f}")
            
            # Interpretation
            if prediction >= 0.92:
                st.success("✅ Excellent packing efficiency! Above target.")
            elif prediction >= 0.88:
                st.info("ℹ️ Good packing efficiency. Within acceptable range.")
            else:
                st.warning("⚠️ Below average efficiency. Consider investigating factors.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE 2: Batch Prediction
# ============================================================================
elif page == "📊 Batch Prediction":
    st.title("📊 Batch %Pack Prediction")
    st.markdown("Upload CSV/Excel file for batch predictions")
    
    uploaded_file = st.file_uploader("Upload File (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df_batch)} records")
            st.dataframe(df_batch.head(10))
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    # Process date features if FullDate exists
                    if 'FullDate' in df_batch.columns:
                        df_batch['FullDate'] = pd.to_datetime(df_batch['FullDate'])
                        df_batch['Year'] = df_batch['FullDate'].dt.year
                        df_batch['Month'] = df_batch['FullDate'].dt.month
                        df_batch['Day'] = df_batch['FullDate'].dt.day
                        df_batch['DayOfWeek'] = df_batch['FullDate'].dt.dayofweek
                        df_batch['Quarter'] = df_batch['FullDate'].dt.quarter
                    
                    # Select and prepare features
                    X_batch = df_batch[feature_names].copy()
                    
                    # Encode categorical
                    for col in categorical_cols:
                        X_batch[col] = label_encoders[col].transform(X_batch[col].astype(str))
                    
                    # Scale
                    X_batch_scaled = scaler.transform(X_batch)
                    
                    # Predict
                    predictions = model.predict(X_batch_scaled)
                    df_batch['Predicted_%Pack'] = predictions
                    df_batch['Predicted_%Pack_Pct'] = predictions * 100
                    
                    st.success("✅ Predictions completed!")
                    
                    # Show results
                    st.dataframe(df_batch[['Predicted_%Pack', 'Predicted_%Pack_Pct']].head(20))
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{predictions.mean():.4f}")
                    col2.metric("Median", f"{np.median(predictions):.4f}")
                    col3.metric("Min", f"{predictions.min():.4f}")
                    col4.metric("Max", f"{predictions.max():.4f}")
                    
                    # Download button
                    csv = df_batch.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="pack_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE 3: Model Analytics
# ============================================================================
elif page == "📈 Model Analytics":
    st.title("📈 Model Performance & Analytics")
    
    tab1, tab2 = st.tabs(["📊 Model Metrics", "🖼️ Visualizations"])
    
    with tab1:
        st.subheader("Model Comparison")
        
        # Load summary
        try:
            summary_df = pd.read_excel('model_summary.xlsx', sheet_name='Model_Comparison')
            st.dataframe(summary_df.style.highlight_max(axis=0, subset=['Test_R2'], color='lightgreen'))
            
            # Best model highlight
            st.success(f"🏆 Best Model: **{best_model_name}** with Test R² = **{metrics['r2_test']:.4f}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Cross-Val R²", f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            col2.metric("RMSE", f"{metrics['rmse']:.4f}")
            col3.metric("MAE", f"{metrics['mae']:.4f}")
            
        except Exception as e:
            st.warning(f"Summary file not found: {e}")
    
    with tab2:
        st.subheader("Model Visualizations")
        
        # Display charts
        viz_files = [
            "01_scatter_actual_vs_predicted.png",
            "02_feature_importance.png",
            "03_shap_summary.png",
            "04_pareto_chart.png",
            "05_correlation_matrix.png",
            "06_residual_plot.png",
            "07_residual_histogram.png",
            "08_main_effect_plot.png",
            "09_interaction_plot.png"
        ]
        
        viz_titles = [
            "Actual vs Predicted",
            "Feature Importance",
            "SHAP Summary",
            "Pareto Chart",
            "Correlation Matrix",
            "Residual Plot",
            "Residual Histogram",
            "Main Effect Plot",
            "Interaction Plot"
        ]
        
        for viz_file, viz_title in zip(viz_files, viz_titles):
            try:
                st.image(viz_file, caption=viz_title, use_container_width=True)
            except:
                st.warning(f"Chart not found: {viz_file}")

# ============================================================================
# PAGE 4: About
# ============================================================================
else:
    st.title("ℹ️ About This Application")
    
    st.markdown("""
    ### 📦 Pack SGA %Pack Predictor
    
    **Purpose:** Predict packing efficiency (%Pack) based on production parameters
    
    **Model Details:**
    - **Best Model:** {0}
    - **Test R² Score:** {1:.4f}
    - **Features:** {2}
    - **Training Samples:** 7,059
    - **Test Samples:** 1,765
    
    **Key Features:**
    - Single prediction for real-time estimation
    - Batch prediction for large-scale analysis
    - Interactive visualizations (9 charts)
    - Model performance metrics
    
    **Created by:** Kyoko (MIT USA)
    **Framework:** Streamlit + Scikit-learn
    **Version:** 1.0
    
    ---
    💡 **Usage Tips:**
    1. Use single prediction for quick estimates
    2. Upload batch files with same column structure as training data
    3. Check analytics for model insights
    4. Download predictions for further analysis
    
    🔒 **Data Security:** All predictions run locally - no data sent to external servers
    """.format(best_model_name, metrics['r2_test'], len(feature_names)))
    
    st.markdown("---")
    st.info("📧 For support or questions, contact your data science team")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
st.sidebar.markdown(f"✓ {best_model_name}")
st.sidebar.markdown(f"✓ R² = {metrics['r2_test']:.4f}")
st.sidebar.markdown(f"✓ Features: {len(feature_names)}")
