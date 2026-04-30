import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

#Absolute path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

#Page configuration
st.set_page_config(
    page_title='Spending Behavior Segmentation',
    layout='wide'
)

#Global CSS
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

#Cached data and model loaders
@st.cache_data
def load_data():
    """Load the clustered dataset produced by kmeans.py."""
    return pd.read_csv(os.path.join(DATA_DIR, 'clustered_data.csv'))

@st.cache_resource
def load_models():
    """Load the three sklearn artefacts saved by kmeans.py."""
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    km = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    return scaler, pca, km

@st.cache_data
def load_cluster_names():
    """Load cluster names from JSON and convert string keys back to int."""
    with open(os.path.join(MODEL_DIR, 'cluster_names.json'), 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

df = load_data()
scaler_model, pca_model, km_model = load_models()
cluster_names = load_cluster_names()

#Colour and label mappings
NAME_TO_COLOR = {
    'High Risk - Cash Advance & Debt Users' : '#ef4444',
    'Premium High Spenders' : '#f59e0b',
    'Active Moderate Spenders' : '#3b82f6',
    'Revolvers - Minimum Payers' : '#f97316',
    'Inactive / Dormant Users' : '#22c55e',
}

NAME_TO_LABEL = {
    'High Risk - Cash Advance & Debt Users' : '[Red]',
    'Premium High Spenders' : '[Yellow]',
    'Active Moderate Spenders' : '[Blue]',
    'Revolvers - Minimum Payers' : '[Orange]',
    'Inactive / Dormant Users' : '[Green]',
}

cluster_hex = {cid: NAME_TO_COLOR.get(name, '#ffffff') for cid, name in cluster_names.items()}
cluster_labels = {cid: NAME_TO_LABEL.get(name, '') for cid, name in cluster_names.items()}

df['Segment'] = df['Cluster'].map(cluster_names)

#Page header
st.markdown('## AI-Based Spending Behavior Segmentation')
st.markdown('*Credit Card Customer Data — K-Means Clustering — 5 Segments*')
st.markdown('---')

tab1, tab2, tab3 = st.tabs(['Overview', 'Cluster Explorer', 'Customer Lookup'])

#TAB 1
with tab1:
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    for col, value, label in zip(
        [col1, col2, col3, col4, col5, col6],
        ['8,950', '17', '5', '0.242', '0.217', '0.169'],
        ['Total Customers', 'Features', 'Clusters',
         'K-Means Silhouette', 'Agglomerative Silhouette', 'BIRCH Silhouette']
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">Segment Distribution</div>',
                    unsafe_allow_html=True)

        counts = df['Cluster'].value_counts().sort_index()

        summary = pd.DataFrame({
            'Segment' : [cluster_labels[i] + ' ' + cluster_names[i] for i in counts.index],
            'Customers' : counts.values,
            'Percentage' : [f'{v / len(df) * 100:.1f}%' for v in counts.values],
        })

        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-title">Segment Split</div>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=[cluster_names[i].split('-')[0].strip() for i in counts.index],
            colors=[cluster_hex[i] for i in counts.index],
            autopct='%1.1f%%',
            textprops={'color': 'white', 'fontsize': 8},
            wedgeprops={'linewidth': 2, 'edgecolor': '#0e1117'},
        )

        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(8)

        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    st.markdown('<div class="section-title">Raw Data Sample</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

#TAB 2
with tab2:
    selected = st.selectbox('Choose a Segment:', options=list(cluster_names.values()))
    cluster_id = [k for k, v in cluster_names.items() if v == selected][0]
    cluster_df = df[df['Cluster'] == cluster_id]

    st.markdown('<br>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    metrics = [
        ('Customers', f'{len(cluster_df):,}'),
        ('Avg Balance', f'Rs {cluster_df["BALANCE"].mean():,.0f}'),
        ('Avg Purchases', f'Rs {cluster_df["PURCHASES"].mean():,.0f}'),
        ('Cash Advance', f'Rs {cluster_df["CASH_ADVANCE"].mean():,.0f}'),
        ('Credit Limit', f'Rs {cluster_df["CREDIT_LIMIT"].mean():,.0f}'),
        ('Full Pay %', f'{cluster_df["PRC_FULL_PAYMENT"].mean() * 100:.1f}%'),
    ]

    for col, (label, value) in zip([c1, c2, c3, c4, c5, c6], metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Segment vs Overall Average</div>',
                unsafe_allow_html=True)

    key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    seg_avg = cluster_df[key_cols].mean()
    overall_avg = df[key_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')

    x = range(len(key_cols))
    ax.bar([i - 0.2 for i in x], seg_avg.values,width=0.4, label='Segment Avg', color=cluster_hex[cluster_id], alpha=0.9)
    ax.bar([i + 0.2 for i in x], overall_avg.values,width=0.4, label='Overall Avg', color='#6b7280', alpha=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(key_cols, color='white', rotation=10)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#3d4470')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    ax.set_title(f'{selected} vs Overall Average', color='white', fontsize=12, pad=15)

    st.pyplot(fig)
    plt.close()

    st.markdown('---')
    st.markdown('<div class="section-title">Segment Sample Data</div>',
                unsafe_allow_html=True)
    st.dataframe(cluster_df.head(8), use_container_width=True)

#TAB 3
with tab3:
    st.markdown('<div class="section-title">Lookup by Customer Index</div>',
                unsafe_allow_html=True)

    customer_idx = st.number_input(
        'Enter Customer Index (0 to 8949):',
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    customer = df.iloc[customer_idx]
    cluster_id = int(customer['Cluster'])

    st.markdown(f'### {cluster_labels[cluster_id]} Customer #{customer_idx} belongs to:')
    st.markdown(f'## {cluster_names[cluster_id]}')
    st.markdown('---')

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    lookup_metrics = [
        ('Balance', f'Rs {customer["BALANCE"]:,.0f}'),
        ('Purchases', f'Rs {customer["PURCHASES"]:,.0f}'),
        ('Cash Advance', f'Rs {customer["CASH_ADVANCE"]:,.0f}'),
        ('Credit Limit', f'Rs {customer["CREDIT_LIMIT"]:,.0f}'),
        ('Payments', f'Rs {customer["PAYMENTS"]:,.0f}'),
        ('Full Pay %', f'{customer["PRC_FULL_PAYMENT"] * 100:.1f}%'),
    ]

    for col, (label, value) in zip([c1, c2, c3, c4, c5, c6], lookup_metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('---')

    st.markdown('<div class="section-title">Test With Custom Values</div>',
                unsafe_allow_html=True)
    st.markdown('Enter any customer values to predict their segment:')

    st.markdown('**Balance and Payments**')
    col1, col2 = st.columns(2)

    with col1:
        balance = st.number_input('Balance (Rs)', min_value=0.0, max_value=20000.0, value=1000.0, step=100.0)
        payments = st.number_input('Payments (Rs)', min_value=0.0, max_value=50000.0, value=1000.0, step=100.0)
        minimum_payments = st.number_input('Minimum Payments (Rs)', min_value=0.0, max_value=30000.0, value=200.0, step=100.0)
    with col2:
        balance_freq = st.number_input('Balance Frequency', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        prc_full = st.number_input('Full Payment Ratio', min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    st.markdown('**Purchases**')
    col3, col4 = st.columns(2)

    with col3:
        purchases = st.number_input('Purchases (Rs)', min_value=0.0, max_value=50000.0, value=500.0, step=100.0)
        purchases_freq = st.number_input('Purchases Frequency', min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    with col4:
        oneoff_freq = st.number_input('One-off Purchases Frequency', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        installments_freq = st.number_input('Installments Purchases Frequency', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.markdown('**Credit and Cash Advance**')
    col5, col6 = st.columns(2)

    with col5:
        credit_limit = st.number_input('Credit Limit (Rs)', min_value=0.0, max_value=30000.0, value=5000.0, step=500.0)
        cash_advance = st.number_input('Cash Advance (Rs)', min_value=0.0, max_value=47000.0, value=0.0, step=100.0)
    with col6:
        cash_adv_freq = st.number_input('Cash Advance Frequency', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        tenure = st.number_input('Tenure (months)', min_value=1, max_value=12, value=10, step=1)

    if st.button('Predict Segment', type='primary'):

        new_customer = {
            'BALANCE' : balance,
            'BALANCE_FREQUENCY' : balance_freq,
            'PURCHASES' : purchases,
            'PURCHASES_FREQUENCY' : purchases_freq,
            'ONEOFF_PURCHASES_FREQUENCY' : oneoff_freq,
            'PURCHASES_INSTALLMENTS_FREQUENCY' : installments_freq,
            'CASH_ADVANCE' : cash_advance,
            'CASH_ADVANCE_FREQUENCY' : cash_adv_freq,
            'CREDIT_LIMIT' : credit_limit,
            'PAYMENTS' : payments,
            'MINIMUM_PAYMENTS' : minimum_payments,
            'PRC_FULL_PAYMENT' : prc_full,
            'TENURE' : int(tenure),
        }

        used_features = scaler_model.feature_names_in_.tolist()

        new_df = pd.DataFrame([new_customer], columns=used_features)
        new_scaled = scaler_model.transform(new_df)
        new_pca = pca_model.transform(new_scaled)
        predicted = int(km_model.predict(new_pca)[0])

        st.success(f'**Predicted Segment:** {cluster_names[predicted]}')
        st.markdown(f'### {cluster_labels[predicted]} {cluster_names[predicted]}')

        key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE','CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']

        clustered = pd.read_csv(os.path.join(DATA_DIR, 'clustered_data.csv'))
        profile = clustered.groupby('Cluster')[key_cols].mean().round(2)

        compare = pd.DataFrame({
            'Feature' : key_cols,
            'Your Customer' : [new_customer[c] for c in key_cols],
            'Segment Average' : [profile.loc[predicted, c] for c in key_cols],
        })

        st.dataframe(compare, use_container_width=True, hide_index=True)