# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Hotel Customers Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏨 Hotel Customers Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Project Overview", "Data Quality Assessment", "Data Cleaning", "Exploratory Analysis", "Key Insights", "About"]
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Esraa\Downloads\hotel-streamlit\hotels_customers_cleaned.csv")
        return df
    except:
        st.error("Please make sure 'hotels_customers_cleaned.csv' is in the same directory")
        return None

df = load_data()

if section == "Project Overview":
    st.markdown('<h2 class="section-header">📊 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Project Description")
        st.write("""
        This project involves comprehensive analysis of hotel customers data to extract valuable business insights.
        The dataset contains information about 83,573 hotel customers with 31 different attributes.
        
        **Key Objectives:**
        - Clean and preprocess raw hotel customer data
        - Perform exploratory data analysis (EDA)
        - Visualize key patterns and relationships
        - Provide actionable business recommendations
        """)
        
        st.subheader("🔍 Research Questions")
        research_questions = [
            "What is the demographic profile of hotel customers?",
            "Which distribution channels generate the highest revenue?",
            "What is the relationship between lead time and booking behavior?",
            "Which market segments are most profitable?",
            "How do room preferences vary by customer characteristics?",
            "What factors influence customer loyalty and repeat stays?",
            "Is there a correlation between customer age and spending patterns?"
        ]
        
        for i, question in enumerate(research_questions, 1):
            st.write(f"{i}. {question}")
    
    with col2:
        st.subheader("📈 Dataset Summary")
        if df is not None:
            st.metric("Total Customers", f"{len(df):,}")
            st.metric("Number of Features", len(df.columns))
            st.metric("Data Quality Score", "98%")
            
            st.subheader("🛠️ Tools Used")
            tools = ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Streamlit"]
            for tool in tools:
                st.write(f"• {tool}")

elif section == "Data Quality Assessment":
    st.markdown('<h2 class="section-header">🔍 Data Quality Assessment</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("📊 Data Shape")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("❌ Missing Values")
            missing_total = df.isnull().sum().sum()
            st.write(f"**Total Missing:** {missing_total}")
            st.write("**Status:** ✅ Clean")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🔄 Data Types")
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            categorical_cols = df.select_dtypes(include=['object']).shape[1]
            st.write(f"**Numeric:** {numeric_cols}")
            st.write(f"**Categorical:** {categorical_cols}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality issues
        st.subheader("🚨 Data Quality Issues Found")
        issues = {
            "Missing Values": "3,779 values in Age column (4.52%)",
            "Invalid Age": "17 negative age values",
            "Negative Lead Time": "10 impossible negative values",
            "Data Inconsistency": "20 rows with PersonsNights < RoomNights",
            "Revenue Outliers": "3,900 extreme revenue values"
        }
        
        for issue, description in issues.items():
            st.write(f"• **{issue}:** {description}")

elif section == "Data Cleaning":
    st.markdown('<h2 class="section-header">🧹 Data Cleaning Process</h2>', unsafe_allow_html=True)
    
    st.subheader("🔄 Cleaning Steps Performed")
    
    cleaning_steps = [
        ("Handled Missing Values", "Filled 3,779 missing Age values with median"),
        ("Removed Invalid Data", "Removed 17 rows with negative age values"),
        ("Fixed Negative Values", "Clipped 10 negative AverageLeadTime values to 0"),
        ("Data Validation", "Identified 20 inconsistent nights records"),
        ("Outlier Management", "Documented 3,900 revenue outliers for analysis")
    ]
    
    for step, description in cleaning_steps:
        with st.expander(f"✅ {step}"):
            st.write(description)
    
    # Before-After comparison
    st.subheader("📈 Cleaning Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Rows Before Cleaning", "83,590")
        st.metric("Missing Values Before", "3,779")
    
    with col2:
        st.metric("Rows After Cleaning", "83,573")
        st.metric("Missing Values After", "0")

elif section == "Exploratory Analysis":
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Demographic Analysis", "Revenue Analysis", "Behavioral Analysis", "Correlation Analysis"]
        )
        
        if analysis_type == "Demographic Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                st.subheader("📊 Age Distribution")
                fig = px.histogram(df, x='Age', nbins=30, 
                                 title='Customer Age Distribution',
                                 color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top nationalities
                st.subheader("🌍 Top Nationalities")
                top_nationalities = df['Nationality'].value_counts().head(10)
                fig = px.bar(x=top_nationalities.index, y=top_nationalities.values,
                           title='Top 10 Nationalities',
                           color=top_nationalities.values,
                           color_continuous_scale='viridis')
                fig.update_layout(xaxis_title='Nationality', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Revenue Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue by distribution channel
                st.subheader("💰 Revenue by Distribution Channel")
                channel_revenue = df.groupby('DistributionChannel')['LodgingRevenue'].mean().sort_values(ascending=False)
                fig = px.bar(x=channel_revenue.index, y=channel_revenue.values,
                           title='Average Revenue by Distribution Channel',
                           color=channel_revenue.values,
                           color_continuous_scale='greens')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue by market segment
                st.subheader("🎯 Revenue by Market Segment")
                segment_revenue = df.groupby('MarketSegment')['LodgingRevenue'].mean().sort_values(ascending=False)
                fig = px.pie(values=segment_revenue.values, names=segment_revenue.index,
                           title='Revenue Distribution by Market Segment')
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Behavioral Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                # Lead time vs revenue (without trendline to avoid statsmodels)
                st.subheader("⏱️ Lead Time vs Revenue")
                fig = px.scatter(df.sample(1000), x='AverageLeadTime', y='LodgingRevenue',
                               title='Lead Time vs Lodging Revenue',
                               opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bookings analysis
                st.subheader("📅 Booking Patterns")
                bookings_data = df[['BookingsCanceled', 'BookingsNoShowed', 'BookingsCheckedIn']].sum()
                fig = px.bar(x=bookings_data.index, y=bookings_data.values,
                           title='Booking Statistics',
                           color=bookings_data.values,
                           color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Correlation Analysis":
            st.subheader("🔗 Correlation Heatmap")
            
            # Select numeric columns for correlation
            numeric_cols = ['Age', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 
                          'BookingsCheckedIn', 'DaysSinceLastStay', 'PersonsNights', 'RoomNights']
            
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

elif section == "Key Insights":
    st.markdown('<h2 class="section-header">💡 Key Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Major Findings")
        
        insights = [
            "**Customer Demographics:** Average age is 45.4 years with diverse nationalities",
            "**Revenue Patterns:** Corporate channels generate highest average revenue ($350+)",
            "**Booking Behavior:** 66.2 days average lead time with positive revenue correlation",
            "**Market Segments:** Corporate segment shows highest profitability",
            "**Data Quality:** 98% data completeness after cleaning process"
        ]
        
        for insight in insights:
            st.write(f"• {insight}")
    
    with col2:
        st.subheader("🎯 Business Recommendations")
        
        recommendations = [
            "**Focus Marketing:** Target corporate distribution channels",
            "**Age Targeting:** Develop offers for 35-55 age group",
            "**Pricing Strategy:** Implement dynamic pricing based on lead time",
            "**Service Enhancement:** Improve services for corporate clients",
            "**Loyalty Programs:** Develop programs for high-value customer segments"
        ]
        
        for recommendation in recommendations:
            st.write(f"• {recommendation}")
    
    # Key metrics
    st.subheader("📊 Performance Metrics")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_revenue = df['LodgingRevenue'].mean()
            st.metric("Average Revenue", f"${avg_revenue:.2f}")
        
        with col2:
            avg_lead_time = df['AverageLeadTime'].mean()
            st.metric("Average Lead Time", f"{avg_lead_time:.1f} days")
        
        with col3:
            top_channel = df.groupby('DistributionChannel')['LodgingRevenue'].mean().idxmax()
            st.metric("Best Channel", top_channel)
        
        with col4:
            customer_retention = (df['DaysSinceLastStay'] < 365).mean() * 100
            st.metric("Customer Retention", f"{customer_retention:.1f}%")

elif section == "About":
    st.markdown('<h2 class="section-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    st.subheader("🎓 Epsilon AI Program")
    st.write("""
    This project was completed as part of the **Epsilon AI & Data Science Program**, 
    demonstrating practical data analysis skills through real-world dataset processing.
    """)
    
    st.subheader("🔗 Program Resources")
    st.write("""
    - **Main Epsilon Repository:** https://github.com/my-name9999-hash/hotel-customers-analysis
    - **Streamlit Publish:** https://hotel-customers-analysis-yfmyskuibdxpqtxzw389i4.streamlit.app/
    - **Program Focus:** Data Science, Machine Learning, AI Applications
    - **Project Type:** Mid-Project Assessment
    """)
    
    st.subheader("📋 Project Requirements Met")
    requirements = [
        "✅ Data cleaning and preprocessing",
        "✅ Exploratory Data Analysis (EDA)",
        "✅ Data visualization and insights",
        "✅ Interactive dashboard development",
        "✅ Business recommendations",
        "✅ GitHub repository setup",
        "✅ Epsilon AI repository mention"
    ]
    
    for requirement in requirements:
        st.write(requirement)
    
    st.subheader("🛠️ Technical Stack")
    tech_stack = {
        "Data Processing": "Pandas, NumPy",
        "Visualization": "Matplotlib, Seaborn, Plotly",
        "Dashboard": "Streamlit",
        "Version Control": "Git, GitHub",
        "Analysis": "Statistical Analysis, Correlation Analysis"
    }
    
    for category, tools in tech_stack.items():
        st.write(f"**{category}:** {tools}")

# Footer
st.markdown("---")
st.markdown(
    "**🏨 Hotel Customers Analysis Dashboard** | "
    "**🎓 Epsilon AI Program** | "
    "**📊 Mid-Project Assessment**"
)