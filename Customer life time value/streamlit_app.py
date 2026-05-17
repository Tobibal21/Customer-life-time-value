# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Lifetime Value Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💰 Customer Lifetime Value (CLV) Analytics</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Overview", "CLV Analysis", "Customer Segmentation", "RFM Analysis", "Predictive Analytics", "Data Upload"]
)

# Data upload section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Source")

# Sample data generator
@st.cache_data
def generate_sample_data(n_customers=1000):
    np.random.seed(42)
    
    customers = []
    for i in range(n_customers):
        # Customer demographics
        age = np.random.randint(18, 70)
        gender = np.random.choice(['Male', 'Female'])
        location = np.random.choice(['North', 'South', 'East', 'West'])
        
        # Purchase behavior
        n_transactions = np.random.poisson(5) + 1
        avg_order_value = np.random.gamma(2, 50)
        total_spent = n_transactions * avg_order_value
        
        # Recency (days since last purchase)
        recency = np.random.exponential(30)
        
        # Tenure (days since first purchase)
        tenure = np.random.gamma(100, 10)
        
        # CLV calculation
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

# Load data
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data loaded successfully!")
else:
    df = generate_sample_data()
    st.sidebar.info("ℹ️ Using sample data. Upload your own CSV file for custom analysis.")

# Display data info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Dataset Info")
st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(f"**Total Revenue:** ${df['total_spent'].sum():,.2f}")

# Overview Dashboard
if analysis_type == "Overview":
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">${:,.2f}</div>
            <div class="metric-label">Average CLV</div>
        </div>
        """.format(df['clv'].mean()), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">${:,.2f}</div>
            <div class="metric-label">Avg Order Value</div>
        </div>
        """.format(df['avg_order_value'].mean()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Avg Transactions</div>
        </div>
        """.format(df['n_transactions'].mean()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Avg Churn Risk</div>
        </div>
        """.format(df['churn_risk'].mean()), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CLV Distribution")
        fig = px.histogram(df, x='clv', nbins=50, title="Customer Lifetime Value Distribution",
                          color_discrete_sequence=['#1E88E5'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("CLV by Location")
        fig = px.box(df, x='location', y='clv', title="CLV Distribution by Location",
                    color='location', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top customers
    st.subheader("🏆 Top 10 Customers by CLV")
    top_customers = df.nlargest(10, 'clv')[['customer_id', 'clv', 'total_spent', 'n_transactions', 'churn_risk']]
    st.dataframe(top_customers, use_container_width=True)

# CLV Analysis
elif analysis_type == "CLV Analysis":
    st.header("📈 CLV Deep Dive Analysis")
    
    # CLV by segment
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CLV by Gender")
        fig = px.bar(df.groupby('gender')['clv'].mean().reset_index(), 
                    x='gender', y='clv', title="Average CLV by Gender",
                    color='gender', text='clv')
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("CLV vs Age")
        fig = px.scatter(df, x='age', y='clv', title="CLV vs Age Relationship",
                        color='gender', size='total_spent', opacity=0.6,
                        labels={'clv': 'Customer Lifetime Value ($)', 'age': 'Age (years)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation with CLV")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['clv'].sort_values(ascending=False)
    
    fig = px.bar(x=correlations.index, y=correlations.values, 
                title="Correlation with CLV",
                labels={'x': 'Features', 'y': 'Correlation Coefficient'},
                color=correlations.values, color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)
    
    # CLV segments
    st.subheader("CLV Segmentation")
    df['clv_segment'] = pd.qcut(df['clv'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    clv_segment_stats = df.groupby('clv_segment').agg({
        'clv': ['count', 'mean', 'std'],
        'total_spent': 'mean',
        'n_transactions': 'mean'
    }).round(2)
    
    st.dataframe(clv_segment_stats, use_container_width=True)
    
    # Revenue contribution
    st.subheader("Revenue Contribution by Segment")
    segment_revenue = df.groupby('clv_segment')['total_spent'].sum()
    
    fig = px.pie(values=segment_revenue.values, names=segment_revenue.index,
                title="Revenue Distribution by CLV Segment",
                hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

# Customer Segmentation
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
        st.subheader("Customer Segment Distribution")
        segment_counts = df['segment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segments", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Segment Metrics")
        segment_metrics = df.groupby('segment').agg({
            'customer_id': 'count',
            'clv': 'mean',
            'total_spent': 'sum',
            'n_transactions': 'mean'
        }).round(2)
        segment_metrics.columns = ['Customer Count', 'Avg CLV', 'Total Revenue', 'Avg Transactions']
        st.dataframe(segment_metrics, use_container_width=True)
    
    # Segment characteristics
    st.subheader("Segment Characteristics Dashboard")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average CLV by Segment", "Average Transactions by Segment"))
    
    fig.add_trace(
        go.Bar(x=df.groupby('segment')['clv'].mean().index,
               y=df.groupby('segment')['clv'].mean().values,
               name='Avg CLV', marker_color='#1E88E5'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df.groupby('segment')['n_transactions'].mean().index,
               y=df.groupby('segment')['n_transactions'].mean().values,
               name='Avg Transactions', marker_color='#FF6B6B'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer list by segment
    selected_segment = st.selectbox("Select segment to view customers", df['segment'].unique())
    segment_customers = df[df['segment'] == selected_segment][['customer_id', 'clv', 'total_spent', 'n_transactions', 'recency_days']]
    st.dataframe(segment_customers, use_container_width=True)

# RFM Analysis
elif analysis_type == "RFM Analysis":
    st.header("🎯 RFM Analysis (Recency, Frequency, Monetary)")
    
    # RFM distribution
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
elif analysis_type == "Predictive Analytics":
    st.header("🔮 Predictive CLV Analytics")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    
    st.subheader("CLV Prediction Model")
    
    # Prepare data for modeling
    feature_cols = ['age', 'n_transactions', 'avg_order_value', 'recency_days', 'tenure_days']
    X = df[feature_cols]
    y = df['clv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    
    with col2:
        st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
    
    with col3:
        st.metric("RMSE", f"${np.sqrt(mean_absolute_error(y_test, y_pred)):.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                title="Feature Importance for CLV Prediction",
                color='importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs Actual
    st.subheader("Predicted vs Actual CLV")
    comparison_df = pd.DataFrame({
        'Actual': y_test.values[:100],
        'Predicted': y_pred[:100]
    })
    
    fig = px.scatter(comparison_df, x='Actual', y='Predicted', 
                    title="Predicted vs Actual CLV Values",
                    labels={'Actual': 'Actual CLV ($)', 'Predicted': 'Predicted CLV ($)'},
                    trendline='ols')
    fig.add_shape(type='line', x0=0, y0=0, x1=comparison_df.max().max(), y1=comparison_df.max().max(),
                 line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual customer prediction
    st.subheader("Predict CLV for New Customer")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 70, 35)
        n_transactions = st.slider("Expected Number of Transactions", 1, 20, 5)
        avg_order_value = st.slider("Average Order Value ($)", 10, 500, 100)
    
    with col2:
        recency_days = st.slider("Days Since Last Purchase", 0, 365, 30)
        tenure_days = st.slider("Customer Tenure (days)", 0, 1000, 100)
    
    if st.button("Predict CLV"):
        new_customer = np.array([[age, n_transactions, avg_order_value, recency_days, tenure_days]])
        predicted_clv = model.predict(new_customer)[0]
        st.success(f"💰 Predicted Customer Lifetime Value: **${predicted_clv:,.2f}**")
        
        # Show confidence interval
        st.info(f"📊 This prediction is based on {X_train.shape[0]} existing customers with similar characteristics.")

# Data Upload and Exploration
else:
    st.header("📊 Data Exploration")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Data statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        fig = px.bar(x=missing_values.index, y=missing_values.values,
                    title="Missing Values by Column",
                    labels={'x': 'Column', 'y': 'Missing Values Count'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Null Count': df.isnull().sum().values,
        'Sample Values': [str(df[col].iloc[0]) for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Customer Lifetime Value Analytics Dashboard | Built with Streamlit</p>
    <p>📊 Analyze customer segments | 💰 Optimize marketing spend | 🎯 Improve retention strategies</p>
</div>
""", unsafe_allow_html=True)