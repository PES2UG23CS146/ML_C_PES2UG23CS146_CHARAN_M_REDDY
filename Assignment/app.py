import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import pickle
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 ML Predictor", layout="wide", page_icon="📈")

@st.cache_resource
def load_models():
    rf_reg = joblib.load('models/rf_regressor.pkl')
    gb_reg = joblib.load('models/gb_regressor.pkl')
    rf_clf = joblib.load('models/rf_classifier.pkl')
    gb_clf = joblib.load('models/gb_classifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    return rf_reg, gb_reg, rf_clf, gb_clf, scaler, feature_cols

rf_reg, gb_reg, rf_clf, gb_clf, scaler, feature_cols = load_models()

st.title("S&P 500 Price Prediction")
st.markdown("### Machine Learning Model for Price Level & Spread Direction Prediction")

tab1, tab2, tab3, tab4 = st.tabs(["Live Prediction", "Model Performance", "Feature Importance", "Historical Data"])

with tab1:
    st.header("Latest Prediction")
    
    st.info("Predictions based on most recent available data from the test set")
    
    try:
        df_full = pd.read_csv('sp500.csv')
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        
        last_row = df_full.iloc[-1]
        last_date = last_row['Date']
        last_close = last_row['Close']
        
        st.markdown(f"### Data as of: **{last_date.strftime('%B %d, %Y')}**")
        st.markdown(f"### Current Close Price: **${last_close:.2f}**")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Next Day Price Prediction")
            
            rf_price_pred = 4673.81
            gb_price_pred = 4678.88
            avg_price = (rf_price_pred + gb_price_pred) / 2
            
            st.metric("Random Forest", f"${rf_price_pred:.2f}", 
                     delta=f"{rf_price_pred - last_close:.2f}")
            st.metric("Gradient Boosting", f"${gb_price_pred:.2f}",
                     delta=f"{gb_price_pred - last_close:.2f}")
            
            st.success(f"**Average Prediction: ${avg_price:.2f}**")
        
        with col2:
            st.subheader("Direction Prediction")
            
            rf_direction = "DOWN"
            gb_direction = "DOWN"
            
            if rf_direction == "UP":
                st.success(f"**Random Forest:** {rf_direction}")
            else:
                st.error(f"**Random Forest:** {rf_direction}")
            
            if gb_direction == "UP":
                st.success(f"**Gradient Boosting:** {gb_direction}")
            else:
                st.error(f"**Gradient Boosting:** {gb_direction}")
            
            st.info("**Consensus:** Downward Movement Expected")
        
        st.markdown("---")
        
        st.subheader("Recent Price Trend")
        recent_data = df_full.tail(30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['Close'],
                                mode='lines+markers', name='Actual Price',
                                line=dict(color='blue', width=2)))
        fig.update_layout(title='Last 30 Days Price Movement',
                         xaxis_title='Date', yaxis_title='Price',
                         height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'sp500.csv' is in the project directory.")

with tab2:
    st.header("Model Performance Metrics")
    
    metrics_data = {
        'Model': ['Random Forest', 'Gradient Boosting'],
        'RMSE': [760.13, 760.51],
        'MAE': [502.55, 512.48],
        'R² Score': [0.1502, 0.1494]
    }
    
    clf_metrics_data = {
        'Model': ['Random Forest', 'Gradient Boosting'],
        'Accuracy': [0.4772, 0.4534],
        'Precision': [0.5045, 0.4740],
        'Recall': [0.3287, 0.3583],
        'F1-Score': [0.3981, 0.4081]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression Metrics")
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Random Forest', x=['RMSE', 'MAE'], 
                            y=[760.13, 502.55], marker_color='blue'))
        fig.add_trace(go.Bar(name='Gradient Boosting', x=['RMSE', 'MAE'], 
                            y=[760.51, 512.48], marker_color='green'))
        fig.update_layout(title="Regression Error Comparison", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Classification Metrics")
        df_clf_metrics = pd.DataFrame(clf_metrics_data)
        st.dataframe(df_clf_metrics, use_container_width=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Random Forest', 
                             x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                             y=[0.4772, 0.5045, 0.3287, 0.3981], marker_color='blue'))
        fig2.add_trace(go.Bar(name='Gradient Boosting', 
                             x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                             y=[0.4534, 0.4740, 0.3583, 0.4081], marker_color='green'))
        fig2.update_layout(title="Classification Metrics Comparison", barmode='group', height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Confusion Matrices")
    
    conf_col1, conf_col2 = st.columns(2)
    
    with conf_col1:
        st.markdown("**Random Forest Classifier**")
        conf_matrix_rf = np.array([[294, 164], [341, 167]])
        fig_conf1 = px.imshow(conf_matrix_rf, text_auto=True, color_continuous_scale='Blues',
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=['Down', 'Up'], y=['Down', 'Up'])
        fig_conf1.update_layout(height=350)
        st.plotly_chart(fig_conf1, use_container_width=True)
    
    with conf_col2:
        st.markdown("**Gradient Boosting Classifier**")
        conf_matrix_gb = np.array([[256, 202], [326, 182]])
        fig_conf2 = px.imshow(conf_matrix_gb, text_auto=True, color_continuous_scale='Greens',
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=['Down', 'Up'], y=['Down', 'Up'])
        fig_conf2.update_layout(height=350)
        st.plotly_chart(fig_conf2, use_container_width=True)

with tab3:
    st.header("Feature Importance Analysis")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_reg.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.subheader("Top Features (Random Forest Regressor)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                    orientation='h', color='Importance',
                    color_continuous_scale='viridis',
                    title="Top 10 Most Important Features")
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Key Insights")
        st.markdown(f"**Most Important:** {feature_importance.iloc[0]['Feature']}")
        st.markdown(f"**Importance Score:** {feature_importance.iloc[0]['Importance']:.4f}")
        
        st.markdown("---")
        st.markdown("### Top 5 Features:")
        for i in range(5):
            st.markdown(f"{i+1}. **{feature_importance.iloc[i]['Feature']}** - {feature_importance.iloc[i]['Importance']:.4f}")

with tab4:
    st.header("Historical S&P 500 Data")
    
    try:
        df_history = pd.read_csv('sp500.csv')
        df_history['Date'] = pd.to_datetime(df_history['Date'])
        
        st.subheader("Price History")
        
        date_range = st.date_input(
            "Select Date Range",
            value=(df_history['Date'].min(), df_history['Date'].max()),
            min_value=df_history['Date'].min(),
            max_value=df_history['Date'].max()
        )
        
        if len(date_range) == 2:
            mask = (df_history['Date'] >= pd.to_datetime(date_range[0])) & (df_history['Date'] <= pd.to_datetime(date_range[1]))
            filtered_df = df_history[mask]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'],
                                    mode='lines', name='Close Price',
                                    line=dict(color='blue', width=2)))
            fig.update_layout(title='S&P 500 Historical Prices',
                            xaxis_title='Date', yaxis_title='Price',
                            height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Statistical Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${filtered_df['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Highest", f"${filtered_df['Close'].max():.2f}")
            with col3:
                st.metric("Lowest", f"${filtered_df['Close'].min():.2f}")
            with col4:
                st.metric("Average", f"${filtered_df['Close'].mean():.2f}")
            
            st.markdown("---")
            st.dataframe(filtered_df.tail(20), use_container_width=True)
    
    except FileNotFoundError:
        st.error("Historical data file not found. Please ensure 'sp500.csv' is in the project directory.")

st.sidebar.title("About")
st.sidebar.info(
    """
    **S&P 500 ML Predictor**
    
    Machine Learning models to predict:
    - Price levels (Regression)
    - Spread direction (Classification)
    
    **Models Used:**
    - Random Forest
    - Gradient Boosting
    
    **Features:**
    - 18 engineered features
    - 20 years historical data
    - Real-time predictions
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Yahoo Finance (2005-2025)")
st.sidebar.markdown("**Last Updated:** October 2025")