import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(
    page_title="Spending Behavior Segmentation",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
 
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4470;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #7c9ef8;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3d4470;
    }

    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
        color: #9ca3af;
        padding: 10px 20px;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #7c9ef8;
        border-bottom: 2px solid #7c9ef8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('data/clustered_data.csv')

df = load_data()

cluster_names = {
    0: 'High Risk — Cash Advance & Debt Users',
    1: 'Premium High Spenders',
    2: 'Active Moderate Spenders',
    3: 'Revolvers — Minimum Payers',
    4: 'Inactive / Dormant Users'
}

cluster_colors = {
    0: '🔴',
    1: '🟡',
    2: '🔵',
    3: '🟠',
    4: '🟢'
}

cluster_hex = {
    0: '#ef4444',
    1: '#f59e0b',
    2: '#3b82f6',
    3: '#f97316',
    4: '#22c55e'
}

df['Segment'] = df['Cluster'].map(cluster_names)

st.markdown("## AI-Based Spending Behavior Segmentation")
st.markdown("*Credit Card Customer Data · K-Means Clustering · 5 Segments*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "🔍 Cluster Explorer",
    "📊 Visualizations",
    "👤 Customer Lookup"
])

with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">8,950</div>
            <div class="metric-label">Total Customers</div>
        </div>""", unsafe_allow_html=True)
 
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">17</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)
 
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Clusters</div>
        </div>""", unsafe_allow_html=True)
 
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0.193</div>
            <div class="metric-label">Silhouette Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
 
    with col_left:
        st.markdown('<div class="section-title">📋 Segment Distribution</div>',
                    unsafe_allow_html=True)
        counts = df['Cluster'].value_counts().sort_index()
        summary = pd.DataFrame({
            'Segment': [cluster_colors[i] + ' ' + cluster_names[i] for i in counts.index],
            'Customers': counts.values,
            'Percentage': [f"{v/len(df)*100:.1f}%" for v in counts.values]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
 
    with col_right:
        st.markdown('<div class="section-title">📊 Segment Split</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=[cluster_names[i].split('—')[0].strip() for i in counts.index],
            colors=[cluster_hex[i] for i in counts.index],
            autopct='%1.1f%%',
            textprops={'color': 'white', 'fontsize': 8},
            wedgeprops={'linewidth': 2, 'edgecolor': '#0e1117'}
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(8)
        st.pyplot(fig)
 
    st.markdown("---")
    st.markdown('<div class="section-title">Raw Data Sample</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    selected = st.selectbox(
        "Choose a Segment:",
        options=list(cluster_names.values())
    )
    cluster_id = [k for k, v in cluster_names.items() if v == selected][0]
    cluster_df = df[df['Cluster'] == cluster_id]
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    metrics = [
        ("👥 Customers", f"{len(cluster_df):,}"),
        ("💰 Avg Balance", f"₹{cluster_df['BALANCE'].mean():,.0f}"),
        ("🛒 Avg Purchases", f"₹{cluster_df['PURCHASES'].mean():,.0f}"),
        ("💸 Cash Advance", f"₹{cluster_df['CASH_ADVANCE'].mean():,.0f}"),
        ("💳 Credit Limit", f"₹{cluster_df['CREDIT_LIMIT'].mean():,.0f}"),
        ("✅ Full Pay %", f"{cluster_df['PRC_FULL_PAYMENT'].mean()*100:.1f}%")
    ]

    for col, (label, value) in zip([c1,c2,c3,c4,c5,c6], metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Segment vs Overall Average</div>',
                unsafe_allow_html=True)

    key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']

    seg_avg = cluster_df[key_cols].mean()
    overall_avg = df[key_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')

    x = range(len(key_cols))

    bars1 = ax.bar(
        [i - 0.2 for i in x],
        seg_avg.values,
        width=0.4,
        label='Segment Avg',
        color=cluster_hex[cluster_id],
        alpha=0.9
    )

    bars2 = ax.bar(
        [i + 0.2 for i in x],
        overall_avg.values,
        width=0.4,
        label='Overall Avg',
        color='#6b7280',
        alpha=0.7
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(key_cols, color='white', rotation=10)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#3d4470')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    ax.set_title(
        f'{selected} vs Overall Average',
        color='white', fontsize=12, pad=15
    )
 
    st.pyplot(fig)
 
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Segment Sample Data</div>',
                unsafe_allow_html=True)
    st.dataframe(cluster_df.head(8), use_container_width=True)

with tab3:

    inner_tab1, inner_tab2, inner_tab3, inner_tab4, inner_tab5, inner_tab6, inner_tab7 = st.tabs([
        "Cluster Scatter",
        "Cluster Sizes",
        "Cluster Heatmap",
        "Elbow Curve",
        "Silhouette Plot",
        "Algorithm Comparison",
        "Comparison Scatter"
    ])

    plots = [
        ('reports/cluster_scatter.png',
         "PCA reduced 17 features to 2D — each color represents a customer segment"),
        ('reports/cluster_sizes.png',
         "Customer count per segment — Active Moderate is the largest group"),
        ('reports/cluster_heatmap.png',
         "Average feature values per cluster — darker = higher value"),
        ('reports/elbow_curve.png',
         "Elbow method — optimal K=5 selected based on inertia drop"),
        ('reports/silhouette_plot.png',
         "Silhouette score = 0.193 — moderate separation, acceptable for real world data"),
        ('reports/algorithm_comparison.png',
         "K-Means (0.193) outperforms Agglomerative (0.176) — K-Means selected as final algorithm"),
        ('reports/comparison_plot.png',
         "Visual comparison of K-Means vs Agglomerative cluster boundaries using PCA 2D projection")
    ]

    for tab, (path, caption) in zip(
        [inner_tab1, inner_tab2, inner_tab3, inner_tab4, inner_tab5, inner_tab6, inner_tab7],
        plots
    ):
        with tab:
            st.image(path, use_container_width=True)
            st.caption(caption)

with tab4:

    customer_idx = st.number_input(
        "Enter Customer Index (0 to 8949):",
        min_value=0,
        max_value=len(df)-1,
        value=0,
        step=1
    )

    customer = df.iloc[customer_idx]

    cluster_id = int(customer['Cluster'])

    st.markdown(f"### {cluster_colors[cluster_id]} Customer #{customer_idx} belongs to:")
    st.markdown(f"## {cluster_names[cluster_id]}")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
 
    lookup_metrics = [
        ("💰 Balance", f"₹{customer['BALANCE']:,.0f}"),
        ("🛒 Purchases", f"₹{customer['PURCHASES']:,.0f}"),
        ("💸 Cash Advance", f"₹{customer['CASH_ADVANCE']:,.0f}"),
        ("💳 Credit Limit", f"₹{customer['CREDIT_LIMIT']:,.0f}"),
        ("💵 Payments", f"₹{customer['PAYMENTS']:,.0f}"),
        ("✅ Full Pay %", f"{customer['PRC_FULL_PAYMENT']*100:.1f}%")
    ]

    for col, (label, value) in zip([c1,c2,c3,c4,c5,c6], lookup_metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)