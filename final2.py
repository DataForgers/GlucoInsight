import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from streamlit_lottie import st_lottie
import requests
import time
import json

        # Set page configuration
st.set_page_config(
            page_title="GlucoInsight",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for styling
st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 0.5rem;
                font-weight: 700;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #34495e;
                margin-bottom: 1rem;
                text-align: center;
                font-weight: 500;
            }
            .result-box {
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }
            .success-box {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .warning-box {
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                color: #856404;
            }
            .danger-box {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .metric-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .stProgress > div > div > div > div {
                background-color: #3498db;
            }
            .tab-content {
                padding: 20px;
                border-radius: 0 0 10px 10px;
                background-color: #ffffff;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #f1f3f5;
                border-radius: 5px 5px 0 0;
                padding: 10px 20px;
                font-weight: 500;
            }
            .stTabs [aria-selected="true"] {
                background-color: #3498db !important;
                color: white !important;
            }
            .styled-button {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: 500;
                border: none;
                cursor: pointer;
            }
            .divider {
                height: 3px;
                background-color: #e0e0e0;
                margin: 20px 0;
                border-radius: 2px;
            }
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            .sidebar-info {
                padding: 15px;
                background-color: #e8f4f8;
                border-radius: 5px;
                margin-bottom: 15px;
            }
            footer {
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

# Load the trained models
@st.cache_resource
def load_models():
            try:
                model1 = joblib.load('diabetes_symptoms_model.pkl')
                model2 = joblib.load('diabetes_xgb_model.pkl')
                scaler = joblib.load('scaler2.pkl')
                return model1, model2, scaler
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return None, None, None

model1, model2, scaler = load_models()

        # Sidebar for navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>GlucoInsight</h2>", unsafe_allow_html=True)

    # About section (detailed)
    st.markdown("""
    An AI-powered platform that uses machine learning for early diabetes risk screening and stage assessment.

    ### 🧠 Two-Model Approach
    - **Symptom-Based Model**: Assesses reported symptoms to suggest likely diabetes stage.
    - **Health-Indicator Model**: Evaluates medical and lifestyle factors to predict overall risk.

    ⚠️ *Note: GlucoInsight is a screening tool and does not replace professional medical advice.*
    """)

    # User guide
    with st.expander("📘 How to Use"):
        st.write("1. Select a model tab")
        st.write("2. Enter the required data")
        st.write("3. Click 'Predict'")
        st.write("4. Use 'Compare Models' for deeper insight")

    # Privacy notice
    with st.expander("🔒 Data Privacy"):
        st.write("All data entered is processed locally and is not stored or transmitted elsewhere.")


        # Main content
st.markdown("<h1 class='main-header'>GlucoInsight: Smart Platform for Diabetes Detection & Risk Stratification</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Powered by machine learning for accurate, dual-model diabetes assessment</p>", unsafe_allow_html=True)

        # Animated progress bar for visual appeal
with st.container():
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
        # Tabs with enhanced styling
tab1, tab2, tab3, tab4 = st.tabs([
            "Symptom-Based Assessment", 
            "Risk Factor Prediction", 
            "Compare Models",
            "About Diabetes"
        ])

        # Tab 1: Diabetes Staging Model
with tab1:
            #st.markdown("<div class='tab-content'>", unsafe_allow_html=True)

            #st.markdown("<div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.markdown("**About This Model**")
            st.write("This model analyzes common diabetes symptoms to provide an initial assessment of diabetes likelihood.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            col1 = st.columns([1])[0]
            
            with col1:
                st.subheader("Symptom-Based Diabetes")
                st.write("Enter your symptoms and demographic information for an initial diabetes stage assessment.")
                
                # Create two columns for a cleaner layout
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    age = st.number_input("Age", min_value=1, max_value=120, step=1, format="%d", help="Enter your age in years")
                    gender = st.selectbox("Gender", ["", "Male", "Female"], help="Select your gender")
                    polyuria = st.selectbox("Excessive Urination (Polyuria)", ["", "Yes", "No"], 
                                        help="Frequent urination - passing large amounts of urine")
                    polydipsia = st.selectbox("Excessive Thirst (Polydipsia)", ["", "Yes", "No"], 
                                            help="Feeling unusually thirsty all the time")
                
                with col1_2:
                    sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["", "Yes", "No"], 
                                                help="Unexplained decrease in body weight")
                    polyphagia = st.selectbox("Excessive Hunger (Polyphagia)", ["", "Yes", "No"], 
                                        help="Feeling unusually hungry despite eating")
                    irritability = st.selectbox("Irritability", ["", "Yes", "No"], 
                                            help="Feeling easily annoyed or agitated")
                    partial_paresis = st.selectbox("Partial Muscle Weakness (Paresis)", ["", "Yes", "No"], 
                                            help="Weakness affecting specific muscles")
            
            
            
            # Function to preprocess inputs for Model 1
            def preprocess_inputs():
                if not age or not gender or not polyuria or not polydipsia or not sudden_weight_loss or not polyphagia or not irritability or not partial_paresis:
                    st.warning("⚠️ Please fill in all fields before predicting.")
                    return None

                gender_val = 1 if gender == "Male" else 0
                polyuria_val = 1 if polyuria == "Yes" else 0
                polydipsia_val = 1 if polydipsia == "Yes" else 0
                sudden_weight_loss_val = 1 if sudden_weight_loss == "Yes" else 0
                polyphagia_val = 1 if polyphagia == "Yes" else 0
                irritability_val = 1 if irritability == "Yes" else 0
                partial_paresis_val = 1 if partial_paresis == "Yes" else 0

                return np.array([[age / 120, gender_val, polyuria_val, polydipsia_val, sudden_weight_loss_val, 
                                polyphagia_val, irritability_val, partial_paresis_val]])

            # Prediction button with loading animation
            col_button, col_space = st.columns([1, 3])
            with col_button:
                predict_button1 = st.button("🔍 Analyze Symptoms", key="detect_model1")
            
            if predict_button1:
                with st.spinner('Analyzing symptoms...'):
                    time.sleep(1)  # Simulate processing time
                    input_data = preprocess_inputs()
                    
                    if input_data is not None:
                        # Results display
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                        st.subheader("Assessment Results")
                        
                        col_results1, col_results2 = st.columns([2, 2])
                        
                        with col_results1:
                            prediction_proba1 = model1.predict_proba(input_data)
                            diabetes_prob1 = prediction_proba1[0][1]  # Probability of having diabetes for Model 1
                            
                            # Create gauge chart with Plotly
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = diabetes_prob1*100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Diabetes Probability"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 40], 'color': "#c8e6c9"},
                                        {'range': [40, 70], 'color': "#fff9c4"},
                                        {'range': [70, 100], 'color': "#ffccbc"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': diabetes_prob1*100
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_results2:
                            # Decision Making for Model 1
                            if diabetes_prob1 < 0.4:
                                st.markdown(f"""
                                <div class='result-box success-box'>
                                    <h3>✅ Low Risk Detected</h3>
                                    <p>Based on your symptoms, there's a low probability ({diabetes_prob1*100:.1f}%) of diabetes.</p>
                                    <p><b>Recommendation:</b> Maintain a healthy lifestyle and regular check-ups.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif 0.4 <= diabetes_prob1 < 0.7:
                                st.markdown(f"""
                                <div class='result-box warning-box'>
                                    <h3>⚠️ Moderate Risk Detected</h3>
                                    <p>Your symptoms indicate a moderate risk ({diabetes_prob1*100:.1f}%) of diabetes or pre-diabetes.</p>
                                    <p><b>Recommendation:</b> Schedule a blood glucose test and consult a healthcare professional.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='result-box danger-box'>
                                    <h3>🚨 High Risk Detected</h3>
                                    <p>Your symptoms strongly suggest ({diabetes_prob1*100:.1f}%) diabetes may be present.</p>
                                    <p><b>Recommendation:</b> Seek immediate medical attention for proper diagnosis and treatment.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display key symptoms that influenced the prediction
                            st.subheader("Key Factors in Analysis")
                            symptoms = ["Polyuria", "Polydipsia", "Weight Loss", "Polyphagia", "Irritability", "Partial Paresis"]
                            values = [
                                1 if polyuria == "Yes" else 0,
                                1 if polydipsia == "Yes" else 0,
                                1 if sudden_weight_loss == "Yes" else 0,
                                1 if polyphagia == "Yes" else 0,
                                1 if irritability == "Yes" else 0,
                                1 if partial_paresis == "Yes" else 0
                            ]
                            
                            # Create a horizontal bar chart of symptoms
                            symptom_df = pd.DataFrame({
                                'Symptom': symptoms,
                                'Present': values
                            })
                            
                            # Filter to only show present symptoms
                            present_symptoms = symptom_df[symptom_df['Present'] == 1]['Symptom'].tolist()
                            
                            if present_symptoms:
                                st.write("Present symptoms that influenced the assessment:")
                                for symptom in present_symptoms:
                                    st.markdown(f"• {symptom}")
                            else:
                                st.write("No key diabetes symptoms were reported.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 2: Diabetes Risk Prediction Model
with tab2:

            #st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
                # st_lottie(lottie_analysis, height=250, key="analysis_animation2")
            #st.markdown("<div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.markdown("**About This Model**")
            st.write("This model uses comprehensive health and lifestyle data to predict diabetes risk based on multiple factors.")
            st.markdown("</div>", unsafe_allow_html=True)
            #st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
            
            col1 = st.columns([1])[0]
            
            with col1:
                st.subheader("Risk Factor-Based Diabetes Prediction")
                st.write("Enter health metrics and lifestyle factors for a comprehensive diabetes risk assessment.")
                
                # Create three columns for a comprehensive form layout
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    age_risk = st.number_input("Age", min_value=1, max_value=120, value=30, help="Your age in years")
                    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1, 
                                        help="Weight(kg)/(height(m))²")
                    high_bp = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", 
                                        help="Do you have high blood pressure?")
                    gen_health = st.selectbox("General Health", [1, 2, 3, 4, 5], 
                                        format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x-1], 
                                        help="How would you rate your general health?")
                    smoker = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", 
                                    help="Do you smoke cigarettes?")
                    
                with col1_2:
                    
                    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", 
                                    help="Biological sex")
                    
                #with col1_3:
                    physical_activity = st.selectbox("Regular Physical Activity", [0, 1], 
                                                format_func=lambda x: "Yes" if x == 1 else "No", 
                                                help="Do you engage in regular physical activity?")
                    no_doctor_cost = st.selectbox("Couldn't see doctor due to cost", [0, 1], 
                                            format_func=lambda x: "Yes" if x == 1 else "No", 
                                            help="Have you avoided doctor visits due to cost concerns?")
                    income = st.selectbox("Income Level", list(range(1, 9)), 
                                    format_func=lambda x: f"Level {x} ({9-x} = lowest, 1 = highest)", 
                                    help="Income bracket (8 = lowest, 1 = highest)")
                    education = st.selectbox("Education Level", list(range(1, 7)), 
                                        format_func=lambda x: f"Level {x} (1 = least, 6 = most)", 
                                        help="Level of education completed")
            
            
            
            #st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            # Convert user input into a NumPy array for Model 2
            user_input_risk = np.array([[high_bp, gen_health, no_doctor_cost, age_risk, bmi, smoker, sex, income, education, physical_activity]])
            
            # Transform user input using the pre-fitted scaler for Model 2
            user_input_scaled = scaler.transform(user_input_risk)
            
            # Prediction button with loading animation
            col_button, col_space = st.columns([1, 2])
            with col_button:
                predict_button2 = st.button("🔍 Analyze Risk Factors", key="predict_model2")
            
            if predict_button2:
                with st.spinner('Analyzing risk factors...'):
                    time.sleep(1)  # Simulate processing time
                    
                    # Results display
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.subheader("Risk Assessment Results")
                    
                    col_results1, col_results2 = st.columns([2, 2])
                    
                    with col_results1:
                        # Get Probability for Model 2
                        probability_risk = model2.predict_proba(user_input_scaled)[:, 1]  # Get probability for "Yes"
                        prediction_risk = (probability_risk >= 0.5).astype(int)  # Threshold = 0.5 for Model 2
                        
                        # Create gauge chart with Plotly
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = probability_risk[0]*100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Diabetes Risk Probability"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#c8e6c9"},
                                    {'range': [40, 70], 'color': "#fff9c4"},
                                    {'range': [70, 100], 'color': "#ffccbc"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': probability_risk[0]*100
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_results2:
                        # Decision Making for Model 2
                        if probability_risk[0] < 0.4:
                            st.markdown(f"""
                            <div class='result-box success-box'>
                                <h3>✅ Low Risk Detected</h3>
                                <p>Based on risk factors, there's a low probability ({probability_risk[0]*100:.1f}%) of diabetes.</p>
                                <p><b>Recommendation:</b> Maintain healthy habits and continue regular check-ups.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif 0.4 <= probability_risk[0] < 0.7:
                            st.markdown(f"""
                            <div class='result-box warning-box'>
                                <h3>⚠️ Moderate Risk Detected</h3>
                                <p>Your risk factors indicate a moderate probability ({probability_risk[0]*100:.1f}%) of diabetes.</p>
                                <p><b>Recommendation:</b> Consider lifestyle modifications and consult a healthcare provider.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='result-box danger-box'>
                                <h3>🚨 High Risk Detected</h3>
                                <p>Your risk factors strongly suggest ({probability_risk[0]*100:.1f}%) a high risk for diabetes.</p>
                                <p><b>Recommendation:</b> Seek medical consultation for proper screening and preventive measures.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Key risk factors visualization
                        st.subheader("Key Risk Factors")
                        
                        # Create risk factor visualization
                        risk_factors = []
                        risk_scores = []
                        
                        # BMI risk
                        if bmi >= 30:
                            risk_factors.append("High BMI")
                            risk_scores.append(min(bmi/30, 1.0))
                        
                        # Age risk
                        if age_risk >= 45:
                            risk_factors.append("Age")
                            risk_scores.append(min((age_risk-30)/50, 1.0))
                        
                        # Other binary factors
                        if high_bp == 1:
                            risk_factors.append("High Blood Pressure")
                            risk_scores.append(0.8)
                        
                        if smoker == 1:
                            risk_factors.append("Smoking")
                            risk_scores.append(0.7)
                        
                        if physical_activity == 0:
                            risk_factors.append("Lack of Physical Activity")
                            risk_scores.append(0.65)
                        
                        if gen_health >= 4:
                            risk_factors.append("Poor General Health")
                            risk_scores.append(0.75)
                        
                        # Create horizontal bar chart if risk factors exist
                        if risk_factors:
                            risk_df = pd.DataFrame({
                                'Factor': risk_factors,
                                'Impact': risk_scores
                            })
                            
                            fig = px.bar(risk_df, y='Factor', x='Impact', orientation='h',
                                        color='Impact', color_continuous_scale=['green', 'yellow', 'red'],
                                        title='Impact of Key Risk Factors')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No significant risk factors detected.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 3: Compare Models
with tab3:
            #st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
            st.subheader("Comprehensive Model Comparison")
            #st.write("Compare the predictions from both models for a more complete assessment.")

            st.write("""
                This section brings together predictions from both the Symptom-Based Model and the Risk Factor Model, allowing you to compare results side by side.
                By viewing both assessments together, you can gain a clearer picture of your overall diabetes risk and make more informed decisions.""")
            
            # Button to trigger comparison
            compare_button = st.button("🔍 Compare Both Models")
            
            if compare_button:
                with st.spinner('Generating comprehensive comparison...'):
                    time.sleep(1.5)  # Simulate processing time
                    
                    # Check if all necessary data is available
                    input_data = preprocess_inputs()
                    if input_data is None:
                        st.warning("⚠️ Please fill in all fields in the Symptom-Based Assessment tab before comparing.")
                    else:
                        # Get predictions from both models
                        prediction_proba1 = model1.predict_proba(input_data)
                        diabetes_prob1 = prediction_proba1[0][1]  # Probability for Model 1
                        
                        # Probability for Model 2
                        probability_risk = model2.predict_proba(user_input_scaled)[:, 1]  # Probability for Model 2
                        
                        # Display results in a visually appealing way
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                        
                        # Summary cards in columns
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.metric(label="Symptom Model", value=f"{diabetes_prob1*100:.1f}%")
                            st.progress(float(diabetes_prob1))
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.metric(label="Risk Factor Model", value=f"{probability_risk[0]*100:.1f}%")
                            st.progress(float(probability_risk[0]))
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            # Calculate average score
                            avg_score = (diabetes_prob1 + probability_risk[0]) / 2
                            st.metric(label="Combined Assessment", value=f"{avg_score*100:.1f}%")
                            st.progress(float(avg_score))
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Visualization comparison
                        st.subheader("Visual Comparison")
                        
                        # Bar chart comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=['Symptom Model'],
                            y=[diabetes_prob1*100],
                            name='Symptom-Based',
                            marker_color='royalblue'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=['Risk Factor Model'],
                            y=[probability_risk[0]*100],
                            name='Risk Factor-Based',
                            marker_color='indianred'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=['Combined Assessment'],
                            y=[avg_score*100],
                            name='Combined',
                            marker_color='green'
                        ))
                        
                        fig.update_layout(
                            title='Model Comparison',
                            xaxis_title='Model Type',
                            yaxis_title='Probability (%)',
                            legend_title='Assessment Type',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Comprehensive assessment
                        st.subheader("Comprehensive Assessment")
                        
                        # Determine overall risk category
                        if avg_score < 0.4:
                            risk_category = "Low Risk"
                            color = "green"
                            recommendations = [
                                "Maintain a healthy diet rich in fruits, vegetables, and whole grains",
                                "Engage in regular physical activity (at least 150 minutes per week)",
                                "Continue regular health check-ups with blood glucose screening",
                                "Maintain a healthy weight"
                            ]
                        elif 0.4 <= avg_score < 0.7:
                            risk_category = "Moderate Risk"
                            color = "orange"
                            recommendations = [
                                "Schedule an appointment with a healthcare provider for blood glucose testing",
                                "Consider dietary modifications to reduce sugar and simple carbohydrate intake",
                                "Increase physical activity to at least 30 minutes daily",
                                "Monitor your weight and work toward a healthy BMI",
                                "Consider consulting with a diabetes educator or nutritionist"
                            ]




                        else:
                            risk_category = "High Risk"
                            color = "red"
                            recommendations = [
                                "Seek immediate medical consultation for proper diagnosis",
                                "If diagnosed, work with healthcare providers on a comprehensive management plan",
                                "Monitor blood glucose levels as advised by your doctor",
                                "Make significant lifestyle modifications including diet, exercise, and stress management",
                                "Consider joining a diabetes support group or education program",
                                "Follow medication regimens if prescribed"
                            ]
                        
                        # Display comprehensive assessment result
                        st.markdown(f"""
                            <div class='result-box' style='background-color: {color}20; border: 1px solid {color}; padding: 20px; border-radius: 10px;'>
                            <h3 style='color: {color}; text-align: center;'>Overall Assessment: {risk_category}</h3>
                            <p style='text-align: center;'>Combined probability score: {avg_score*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display personalized recommendations
                        st.subheader("Personalized Recommendations")
                        
                        for i, rec in enumerate(recommendations):
                            st.markdown(f"**{i+1}.** {rec}")
                        
                        # Model agreement analysis
                        st.subheader("Model Agreement Analysis")
                        
                        diff = abs(diabetes_prob1 - probability_risk[0])
                        
                        if diff < 0.1:
                            agreement = "High Agreement"
                            agreement_text = "Both models show very similar risk assessments, increasing confidence in the prediction."
                        elif diff < 0.3:
                            agreement = "Moderate Agreement"
                            agreement_text = "There is some variation between the models, which may reflect different risk aspects."
                        else:
                            agreement = "Low Agreement"
                            agreement_text = "The models show significant differences. Consider the specific factors from each model."
                        
                        st.markdown(f"""
                            <div style='background-color: #f1f8e9; padding: 15px; border-radius: 10px; margin-top: 10px;'>
                            <h4>{agreement}</h4>
                            <p>{agreement_text}</p>
                            <p>Difference between models: {diff*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Generate PDF report option
                        #st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                        #st.subheader("Report Generation")
                        #st.write("You can generate a detailed PDF report with all assessment results for healthcare provider consultation.")
                        
                        #if st.button("📄 Generate Detailed Report"):
                            #st.success("Report generation functionality would be implemented here in a production environment.")
                            #st.info("In a real implementation, this would create a downloadable PDF with all assessment details.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 4: About & Information
with tab4:
            
            # Educational information
            st.subheader("Diabetes Education")
            
            #st.markdown("<h2 class='sub-header'>ℹ️ About Diabetes</h2>", unsafe_allow_html=True)
    
            tabs = st.tabs(["What is Diabetes?", "Risk Factors", "Symptoms", "Prevention"])
    
            with tabs[0]:
                    st.write("""
                    **Diabetes** is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.
                    
                    There are two main types of diabetes:
                    - **Type 1 Diabetes**: The body does not produce insulin. People with type 1 diabetes need daily insulin injections to control their blood glucose levels.
                    - **Type 2 Diabetes**: The body does not use insulin effectively. This is the most common type of diabetes and is largely the result of excess body weight and physical inactivity.
                    """)
    
            with tabs[1]:
                st.write("""
                **Risk factors for Type 2 Diabetes include:**
                - Family history of diabetes
                - Overweight or obesity
                - Physical inactivity
                - Age (risk increases with age)
                - High blood pressure
                - History of gestational diabetes
                - Polycystic ovary syndrome
                - History of heart disease or stroke
                - Certain ethnicities (including African American, Latino, Native American, and Asian American)
                """)
            
            with tabs[2]:
                st.write("""
                **Common symptoms of diabetes include:**
                - Polyuria (frequent urination)
                - Polydipsia (excessive thirst)
                - Polyphagia (excessive hunger)
                - Unexpected weight loss
                - Fatigue
                - Blurred vision
                - Slow-healing wounds
                - Frequent infections
                - Tingling or numbness in hands or feet
                
                Note that many people with Type 2 diabetes may not experience symptoms for years.
                """)
            
            with tabs[3]:
                st.write("""
                **Prevention strategies for Type 2 Diabetes:**
                - Maintain a healthy weight
                - Regular physical activity (at least 30 minutes per day)
                - Healthy diet rich in fruits, vegetables, and whole grains
                - Limit sugar and saturated fat intake
                - Don't smoke
                - Limit alcohol consumption
                - Regular health check-ups
                """)
                    









            




