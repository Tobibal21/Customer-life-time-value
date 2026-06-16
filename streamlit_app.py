# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------

# Page Configuration

# -------------------------------

st.set_page_config(
page_title="Customer Lifetime Value Dashboard",
layout="wide",
initial_sidebar_state="expanded"
)

# -------------------------------

# Minimal Styling

# -------------------------------

st.markdown("""

<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    font-weight: 600;
}
</style>

""", unsafe_allow_html=True)

# -------------------------------

# Title

# -------------------------------

st.title("Customer Lifetime Value Analytics")
st.caption("Customer insights, segmentation, and predictive modeling")

# -------------------------------

# Sidebar

# -------------------------------

st.sidebar.title("Navigation")

analysis_type = st.sidebar.selectbox(
"Select Analysis Type",
["Overview", "CLV Analysis", "Customer Segmentation", "RFM Analysis", "Predictive Analytics", "Data Exploration"]
)

st.sidebar.subheader("Data Source")

# -------------------------------

# Sample Data Generator

# -------------------------------

@st.cache_data
def generate_sample_data(n_customers=1000):
    np.random.seed(42)

    customers = []
    for i in range(n_customers):
        age = np.random.randint(18, 70)
        gender = np.random.choice(['Male', 'Female'])
        location = np.random.choice(['North', 'South', 'East', 'West'])

        n_transactions = np.random.poisson(5) + 1
        avg_order_value = np.random.gamma(2, 50)
        total_spent = n_transactions * avg_order_value

        recency = np.random.exponential(30)
        tenure = np.random.gamma(100, 10)

        clv = total_spent * (1 + 0.1 * np.random.random())

        customers.append({
            'customer_id': f'CUST_{i+1:04d}',
            'age': age,
            'gender': gender,
            'location': location,
            'n_transactions': n_transactions,
            'avg_order_value': avg_order_value,
            'total_spent': total_spent,
            'recency_days': recency,
            'tenure_days': tenure,
            'clv': clv,
            'churn_risk': np.random.random() * 100
        })

    return pd.DataFrame(customers)

# -------------------------------

# Data Loading

# -------------------------------

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data loaded successfully")
else:
    df = generate_sample_data()
    st.sidebar.info("Using sample data. Upload a CSV file for custom analysis.")

# Display data info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Dataset Info")
st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(f"**Total Revenue:** ${df['total_spent'].sum():,.2f}")

# -------------------------------

# Overview

# -------------------------------

if analysis_type == "Overview":
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average CLV", f"${df['clv'].mean():,.2f}")
    col2.metric("Avg Order Value", f"${df['avg_order_value'].mean():,.2f}")
    col3.metric("Avg Transactions", f"{df['n_transactions'].mean():.1f}")
    col4.metric("Avg Churn Risk", f"{df['churn_risk'].mean():.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='clv', nbins=50)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x='location', y='clv')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customers by CLV")
    st.dataframe(df.nlargest(10, 'clv'), use_container_width=True)

# -------------------------------

# CLV Analysis

# -------------------------------

elif analysis_type == "CLV Analysis":
    st.header("CLV Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df.groupby('gender')['clv'].mean().reset_index(),
                     x='gender', y='clv')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='age', y='clv')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------

# Customer Segmentation

# -------------------------------

elif analysis_type == "Customer Segmentation":
    st.header("👥 Customer Segmentation Analysis")
    
    # Create segments
    df['recency_score'] = pd.qcut(df['recency_days'], q=4, labels=['4', '3', '2', '1'])
    df['frequency_score'] = pd.qcut(df['n_transactions'], q=4, labels=['1', '2', '3', '4'])
    df['monetary_score'] = pd.qcut(df['total_spent'], q=4, labels=['1', '2', '3', '4'])
    
    df['rfm_score'] = df['recency_score'].astype(str) + df['frequency_score'].astype(str) + df['monetary_score'].astype(str)
    
    # Segment definition
    def segment_customer(row):
        if row['recency_days'] < 30 and row['n_transactions'] > 3:
            return 'Champions'
        elif row['recency_days'] < 60 and row['n_transactions'] > 2:
            return 'Loyal Customers'
        elif row['recency_days'] < 90:
            return 'Recent Customers'
        elif row['n_transactions'] > 1:
            return 'At Risk'
        else:
            return 'Lost'
    
    df['segment'] = df.apply(segment_customer, axis=1)
    
    # Segment distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df['segment'].value_counts())
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(df.groupby('segment').mean(numeric_only=True))


# -------------------------------

# RFM Analysis

# -------------------------------

elif analysis_type == "RFM Analysis":
    st.header("RFM Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Recency Distribution")
        fig = px.histogram(df, x='recency_days', nbins=30, title="Days Since Last Purchase",
                          color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Frequency Distribution")
        fig = px.histogram(df, x='n_transactions', nbins=20, title="Number of Transactions",
                          color_discrete_sequence=['#4ECDC4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Monetary Distribution")
        fig = px.histogram(df, x='total_spent', nbins=30, title="Total Amount Spent",
                          color_discrete_sequence=['#45B7D1'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM scatter plots
    st.subheader("RFM Relationships")
    fig = make_subplots(rows=1, cols=2)
    
    fig.add_trace(
        go.Scatter(x=df['recency_days'], y=df['total_spent'], mode='markers',
                  marker=dict(size=df['n_transactions']/2, color=df['clv'], colorscale='Viridis'),
                  text=df['customer_id'], name='Recency vs Monetary'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['n_transactions'], y=df['total_spent'], mode='markers',
                  marker=dict(size=df['clv']/100, color=df['recency_days'], colorscale='Plasma'),
                  text=df['customer_id'], name='Frequency vs Monetary'),
        row=1, col=2
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # RFM matrix
    st.subheader("RFM Score Matrix")
    rfm_matrix = pd.crosstab(df['recency_score'], df['frequency_score'], 
                             values=df['monetary_score'], aggfunc='mean').fillna(0)
    
    fig = px.imshow(rfm_matrix, text_auto=True, aspect="auto",
                   title="Average Monetary Value by Recency and Frequency Scores",
                   labels=dict(x="Frequency Score", y="Recency Score", color="Avg Monetary"))
    st.plotly_chart(fig, use_container_width=True)

# Predictive Analytics

# -------------------------------

elif analysis_type == "Predictive Analytics":
    st.header("Predictive Analytics")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    feature_cols = ['age', 'n_transactions', 'avg_order_value', 'recency_days', 'tenure_days']

    X = df[feature_cols]
    y = df['clv']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("R2 Score", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("RMSE", f"${np.sqrt(mean_absolute_error(y_test, y_pred)):.2f}")

    st.subheader("Predict CLV")

    age = st.slider("Age", 18, 70, 30)
    transactions = st.slider("Transactions", 1, 20, 5)
    value = st.slider("Order Value", 10, 500, 100)

    if st.button("Run Prediction"):
        pred = model.predict([[age, transactions, value, 30, 100]])[0]
        st.success(f"Predicted CLV: ${pred:,.2f}")

# -------------------------------

# Data Exploration

# -------------------------------
else:
    st.header("Data Exploration")

    st.dataframe(df.head(100), use_container_width=True)
    st.dataframe(df.describe(), use_container_width=True)

# -------------------------------

# Footer

# -------------------------------

st.markdown("---")
st.caption("Customer Lifetime Value Dashboard | Built with Streamlit")
