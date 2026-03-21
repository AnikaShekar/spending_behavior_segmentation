# ================================================================
# FILE: dashboard/app.py
# PROJECT: AI-Based Spending Behavior Segmentation
# PURPOSE: Streamlit web dashboard to visualize clustering results
# HOW TO RUN: streamlit run dashboard/app.py
# ================================================================
 
# ── IMPORTS ─────────────────────────────────────────────────────
# streamlit  → the web dashboard framework we use to build the UI
# pandas     → for loading and manipulating our CSV data
# matplotlib → for creating custom styled charts
# seaborn    → built on matplotlib, for better looking plots
# PIL/Image  → for handling image files (not used directly but good to have)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
 
 
# ================================================================
# PAGE CONFIGURATION
# ================================================================
# st.set_page_config() MUST be the very first Streamlit command
# If you put any st. command before this, you'll get an error
#
# page_title → text shown in the browser tab at the top
# page_icon  → emoji or image shown next to title in browser tab
# layout     → "wide" uses full screen width
#              "centered" (default) uses narrow centered column
st.set_page_config(
    page_title="Spending Behavior Segmentation",
    page_icon="💳",
    layout="wide"
)
 
 
# ================================================================
# CUSTOM CSS STYLING
# ================================================================
# Streamlit has limited built-in styling
# We use st.markdown() with unsafe_allow_html=True to inject CSS
# unsafe_allow_html=True → allows raw HTML/CSS to be rendered
#                          (False by default for security reasons)
#
# The <style> tag tells the browser: "treat this as CSS rules"
# These CSS classes are then used throughout the app
# by adding class="classname" inside HTML divs
st.markdown("""
<style>
    /* 'main' targets the main content area of Streamlit */
    /* Sets dark background color for the whole page */
    .main { background-color: #0e1117; }
 
    /* 'metric-card' is our custom card style for metric boxes */
    /* linear-gradient creates a smooth color transition background */
    /* 135deg = diagonal direction of gradient */
    /* border-radius = rounds the corners of the card */
    /* padding = space inside the card between content and edges */
    /* border = thin colored outline around the card */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4470;
    }
 
    /* 'metric-value' styles the large number inside each card */
    /* font-size: 2rem = 2x the default font size */
    /* font-weight: bold = makes it thick/heavy */
    /* color: #7c9ef8 = light blue color for the numbers */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #7c9ef8;
    }
 
    /* 'metric-label' styles the small description text below the number */
    /* font-size: 0.85rem = slightly smaller than default */
    /* color: #9ca3af = muted grey color */
    /* margin-top = space between number and label */
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }
 
    /* 'section-title' styles the heading above each section */
    /* border-bottom adds an underline effect to the heading */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3d4470;
    }
 
    /* Targets Streamlit's tab buttons using its internal test ID */
    /* data-testid is Streamlit's internal HTML attribute */
    /* Styles ALL tab buttons in their default (unselected) state */
    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
        color: #9ca3af;
        padding: 10px 20px;
    }
 
    /* Targets ONLY the currently selected/active tab button */
    /* aria-selected="true" is an HTML accessibility attribute */
    /* that Streamlit sets on the active tab automatically */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #7c9ef8;
        border-bottom: 2px solid #7c9ef8;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ================================================================
# LOAD DATA
# ================================================================
# @st.cache_data is a decorator that tells Streamlit to cache results
# A decorator modifies the behavior of the function below it
#
# WITHOUT @st.cache_data:
#   Every time user clicks ANYTHING → entire script reruns → data reloads
#   This makes the app slow
#
# WITH @st.cache_data:
#   Data loads once on first run → stored in memory
#   Next time function is called → returns stored result instantly
#   Much faster — especially for large CSV files
@st.cache_data
def load_data():
    # Read the clustered CSV file we created in kmeans.py
    # This file has all 17 features + a 'Cluster' column (values 0-4)
    # Each row = one customer, 8950 rows total
    return pd.read_csv('data/clustered_data.csv')
 
# Call the function to actually load the data into memory
# df is our main dataframe used throughout the entire dashboard
df = load_data()
 
 
# ================================================================
# CLUSTER METADATA — NAMES, COLORS, HEX CODES
# ================================================================
# These dictionaries map cluster numbers (0-4) to meaningful info
# We derived these names by analyzing cluster_profile in kmeans.py
 
# cluster_names: maps cluster number → business segment description
# These descriptions explain WHAT TYPE of customer is in each cluster
cluster_names = {
    0: 'High Risk — Cash Advance & Debt Users',
    # Cluster 0: high BALANCE (4903), high CASH_ADVANCE (4983), low PRC_FULL_PAYMENT (0.04)
    # These customers withdraw cash from card and barely pay it back
 
    1: 'Premium High Spenders',
    # Cluster 1: highest PURCHASES (7815), highest CREDIT_LIMIT (9769)
    # Small group (395) but highest value customers for the bank
 
    2: 'Active Moderate Spenders',
    # Cluster 2: largest group (3164), moderate spending across all features
    # The "average" customer — nothing extreme
 
    3: 'Revolvers — Minimum Payers',
    # Cluster 3: second largest (3047), very low PRC_FULL_PAYMENT (0.02)
    # They carry debt month to month, rarely pay full amount
 
    4: 'Inactive / Dormant Users'
    # Cluster 4: lowest BALANCE (111) and PURCHASES (335)
    # Card is barely used — bank wants to re-engage these customers
}
 
# cluster_colors: emoji indicators for each cluster
# Used in tables and headings for quick visual identification
cluster_colors = {
    0: '🔴',  # red    → danger/high risk
    1: '🟡',  # yellow → gold/premium
    2: '🔵',  # blue   → calm/moderate
    3: '🟠',  # orange → warning/revolvers
    4: '🟢'   # green  → inactive/dormant
}
 
# cluster_hex: actual color codes for matplotlib charts
# These must match the emoji colors above for visual consistency
# Used in bar charts, pie charts, and scatter plots
cluster_hex = {
    0: '#ef4444',  # red
    1: '#f59e0b',  # amber/yellow
    2: '#3b82f6',  # blue
    3: '#f97316',  # orange
    4: '#22c55e'   # green
}
 
# Add a human-readable 'Segment' column to the dataframe
# df['Cluster'] has numbers like 0,1,2,3,4
# .map(cluster_names) replaces each number with its corresponding name
# e.g. 0 → 'High Risk — Cash Advance & Debt Users'
df['Segment'] = df['Cluster'].map(cluster_names)
 
 
# ================================================================
# APP HEADER
# ================================================================
# st.markdown() renders text with Markdown and/or HTML formatting
# ## creates a large heading (H2 in HTML)
# *text* in markdown = italic text
# --- creates a horizontal dividing line
st.markdown("## 💳 AI-Based Spending Behavior Segmentation")
st.markdown("*Credit Card Customer Data · K-Means Clustering · 5 Segments*")
st.markdown("---")
 
 
# ================================================================
# TOP NAVIGATION TABS
# ================================================================
# st.tabs() creates a horizontal tab bar at the top of the content
# Pass a list of tab label strings → returns tab objects
# Each tab object is used as a context manager with 'with' keyword
# Everything INSIDE 'with tab1:' block appears ONLY in that tab
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",       # Tab 1: summary + metrics + raw data
    "🔍 Cluster Explorer", # Tab 2: select segment, see its profile
    "📊 Visualizations",  # Tab 3: all saved plot images
    "👤 Customer Lookup"  # Tab 4: look up individual customer segment
])
 
 
# ================================================================
# TAB 1 — OVERVIEW
# Purpose: Give a quick summary of the entire project at a glance
# Contains: 4 key metrics, segment table, pie chart, raw data
# ================================================================
with tab1:
 
    # st.columns(4) divides the horizontal space into 4 equal parts
    # Returns 4 column objects — we unpack them into col1, col2, col3, col4
    # Everything inside 'with col1:' appears in the first column only
    col1, col2, col3, col4 = st.columns(4)
 
    # We use custom HTML metric cards instead of Streamlit's built-in st.metric()
    # Reason: st.metric() doesn't support custom colors, gradients, or font sizes
    # The f-string (f"""...""") is a multiline formatted string
    # class="metric-card" applies our CSS styling defined above
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
 
    # Add vertical space between rows
    # <br> is an HTML line break — creates empty space
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Two equal columns for table (left) and pie chart (right)
    # [1, 1] means both columns are equal width
    # [2, 1] would make left column twice as wide as right
    col_left, col_right = st.columns([1, 1])
 
    with col_left:
        st.markdown('<div class="section-title">📋 Segment Distribution</div>',
                    unsafe_allow_html=True)
 
        # value_counts() → counts occurrences of each unique value in 'Cluster' column
        # Returns a Series: {0: 987, 1: 395, 2: 3164, 3: 3047, 4: 1357}
        # sort_index() → sorts by cluster number (0,1,2,3,4) instead of by count
        counts = df['Cluster'].value_counts().sort_index()
 
        # Build a display-friendly DataFrame for the table
        # List comprehension creates emoji+name strings for each cluster
        # f"{v/len(df)*100:.1f}%" calculates and formats percentage to 1 decimal place
        summary = pd.DataFrame({
            'Segment': [cluster_colors[i] + ' ' + cluster_names[i] for i in counts.index],
            'Customers': counts.values,
            'Percentage': [f"{v/len(df)*100:.1f}%" for v in counts.values]
        })
 
        # Display the DataFrame as an interactive table
        # use_container_width=True → table stretches to fill column width
        # hide_index=True → hides the default 0,1,2,3,4 row numbers on the left
        st.dataframe(summary, use_container_width=True, hide_index=True)
 
    with col_right:
        st.markdown('<div class="section-title">📊 Segment Split</div>',
                    unsafe_allow_html=True)
 
        # Create matplotlib figure with dark background
        # figsize=(5,4) → width=5 inches, height=4 inches
        # facecolor → background color of the entire figure canvas
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')  # background of the plot area itself
 
        # ax.pie() draws a pie chart
        # counts.values → actual slice sizes (customer counts)
        # labels → text shown next to each slice
        #   .split('—')[0] takes only text before '—' to keep labels short
        #   .strip() removes extra whitespace
        # colors → hex colors from our cluster_hex dict
        # autopct='%1.1f%%' → shows percentage on each slice (1 decimal place)
        # textprops → styling for all text in the chart
        # wedgeprops → styling for the pie slices themselves
        #   edgecolor adds a dark border between slices for visual separation
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=[cluster_names[i].split('—')[0].strip() for i in counts.index],
            colors=[cluster_hex[i] for i in counts.index],
            autopct='%1.1f%%',
            textprops={'color': 'white', 'fontsize': 8},
            wedgeprops={'linewidth': 2, 'edgecolor': '#0e1117'}
        )
 
        # autotexts = the percentage labels on each slice
        # We set them white and small so they're readable on dark background
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(8)
 
        # st.pyplot(fig) → renders the matplotlib figure in Streamlit
        # Always pass the figure object explicitly to avoid warnings
        st.pyplot(fig)
 
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Raw Data Sample</div>',
                unsafe_allow_html=True)
 
    # df.head(10) returns first 10 rows of the dataframe
    # Gives examiners/users a preview of the actual data
    st.dataframe(df.head(10), use_container_width=True)
 
 
# ================================================================
# TAB 2 — CLUSTER EXPLORER
# Purpose: Let user select any segment and see its detailed profile
# Contains: dropdown, 6 metric cards, comparison bar chart, data table
# ================================================================
with tab2:
 
    # st.selectbox() → dropdown menu widget
    # First argument = label shown above the dropdown
    # options = list of items to show in dropdown
    # list(cluster_names.values()) = ['High Risk...', 'Premium...', etc.]
    # Returns whichever option the user currently has selected
    selected = st.selectbox(
        "Choose a Segment:",
        options=list(cluster_names.values())
    )
 
    # Reverse lookup — we have the name, we need the cluster number
    # cluster_names.items() gives pairs: (0, 'High Risk...'), (1, 'Premium...'), etc.
    # We find the key (k) where the value (v) matches what user selected
    # [0] at the end gets the first (and only) result from the list
    cluster_id = [k for k, v in cluster_names.items() if v == selected][0]
 
    # Filter dataframe to only rows belonging to the selected cluster
    # df['Cluster'] == cluster_id → creates a boolean mask (True/False for each row)
    # df[mask] → returns only the rows where mask is True
    cluster_df = df[df['Cluster'] == cluster_id]
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # 6 metric cards side by side showing key averages for selected segment
    c1, c2, c3, c4, c5, c6 = st.columns(6)
 
    # List of (label, value) tuples for each metric card
    # f-string formatting:
    #   :,   → adds comma separators (e.g. 1000 → 1,000)
    #   :.0f → rounds to 0 decimal places
    #   *100 → converts 0.04 proportion to 4.0 percentage
    metrics = [
        ("👥 Customers", f"{len(cluster_df):,}"),
        ("💰 Avg Balance", f"₹{cluster_df['BALANCE'].mean():,.0f}"),
        ("🛒 Avg Purchases", f"₹{cluster_df['PURCHASES'].mean():,.0f}"),
        ("💸 Cash Advance", f"₹{cluster_df['CASH_ADVANCE'].mean():,.0f}"),
        ("💳 Credit Limit", f"₹{cluster_df['CREDIT_LIMIT'].mean():,.0f}"),
        ("✅ Full Pay %", f"{cluster_df['PRC_FULL_PAYMENT'].mean()*100:.1f}%")
    ]
 
    # zip([c1,c2,c3,c4,c5,c6], metrics) pairs each column with its metric
    # e.g. (c1, ("👥 Customers", "987")), (c2, ("💰 Avg Balance", "₹4,903")), etc.
    # We unpack each pair into col and (label, value)
    for col, (label, value) in zip([c1,c2,c3,c4,c5,c6], metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Segment vs Overall Average Bar Chart ─────────────────────
    # This answers: "How is this segment DIFFERENT from the average customer?"
    # Side-by-side bars: segment avg (colored) vs overall avg (grey)
    st.markdown('<div class="section-title">📊 Segment vs Overall Average</div>',
                unsafe_allow_html=True)
 
    # Only compare these 5 key financial features
    # (not all 17 — too cluttered and less meaningful)
    key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
 
    # .mean() calculates the average of each column
    # seg_avg = averages for ONLY the selected cluster's customers
    # overall_avg = averages for ALL 8950 customers
    seg_avg = cluster_df[key_cols].mean()
    overall_avg = df[key_cols].mean()
 
    # Create figure with dark background matching app theme
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
    ax.set_facecolor('#1e2130')  # slightly lighter than outer background
 
    # range(5) creates [0, 1, 2, 3, 4] — one position per feature
    x = range(len(key_cols))
 
    # Segment bars: shifted LEFT by 0.2 from center position
    # width=0.4 means each bar takes up 0.4 units of space
    # color uses the cluster's specific hex color
    # alpha=0.9 means 90% opaque (slightly transparent)
    bars1 = ax.bar(
        [i - 0.2 for i in x],   # x positions shifted left
        seg_avg.values,           # bar heights = segment averages
        width=0.4,
        label='Segment Avg',
        color=cluster_hex[cluster_id],  # cluster's specific color
        alpha=0.9
    )
 
    # Overall bars: shifted RIGHT by 0.2 from center position
    # Grey color (#6b7280) for overall average — neutral, not associated with any cluster
    bars2 = ax.bar(
        [i + 0.2 for i in x],   # x positions shifted right
        overall_avg.values,       # bar heights = overall averages
        width=0.4,
        label='Overall Avg',
        color='#6b7280',          # neutral grey
        alpha=0.7
    )
 
    # Style the chart axes and labels for dark theme
    ax.set_xticks(list(x))                        # set tick positions
    ax.set_xticklabels(key_cols, color='white', rotation=10)  # feature names, slightly rotated
    ax.tick_params(colors='white')                # y-axis numbers in white
    ax.spines[:].set_color('#3d4470')             # axis border color
    ax.legend(facecolor='#1e2130', labelcolor='white')  # legend with dark background
    ax.set_title(
        f'{selected} vs Overall Average',
        color='white', fontsize=12, pad=15        # pad = space between title and chart
    )
 
    st.pyplot(fig)
 
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Segment Sample Data</div>',
                unsafe_allow_html=True)
 
    # Show first 8 rows of the selected cluster's data
    # Lets user see actual customer rows in this segment
    st.dataframe(cluster_df.head(8), use_container_width=True)
 
 
# ================================================================
# TAB 3 — VISUALIZATIONS
# Purpose: Display all saved plot images from reports/ folder
# Contains: 5 nested tabs, one for each plot
# ================================================================
with tab3:
 
    # Nested tabs INSIDE the Visualizations tab
    # This organizes 5 different plots cleanly
    # Without nested tabs, all plots would stack vertically — mess
    inner_tab1, inner_tab2, inner_tab3, inner_tab4, inner_tab5, inner_tab6, inner_tab7 = st.tabs([
        "Cluster Scatter",
        "Cluster Sizes",
        "Cluster Heatmap",
        "Elbow Curve",
        "Silhouette Plot",
        "Algorithm Comparison",
        "Comparison Scatter"
    ])
 
    # List of (image_path, caption) pairs — one for each inner tab
    # These files were saved to reports/ folder during visualization.py run
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
 
    # zip() pairs each inner_tab object with its corresponding plot info
    # We loop through both simultaneously
    for tab, (path, caption) in zip(
        [inner_tab1, inner_tab2, inner_tab3, inner_tab4, inner_tab5, inner_tab6, inner_tab7],
        plots
    ):
        with tab:
            # st.image() displays an image file
            # use_container_width=True → image stretches to fill tab width
            st.image(path, use_container_width=True)
 
            # st.caption() shows small italic text below the image
            # Used to explain what each plot shows
            st.caption(caption)
 
 
# ================================================================
# TAB 4 — CUSTOMER LOOKUP
# Purpose: Enter any customer row index → see their segment + details
# Useful for viva demo — shows model works on individual level
# ================================================================
with tab4:
 
    # st.number_input() → numeric input box with +/- increment buttons
    # min_value = lowest allowed input (0 = first row)
    # max_value = highest allowed input (8949 = last row, since 0-indexed)
    # value = default value shown when page loads
    # step = how much +/- buttons change the value by
    customer_idx = st.number_input(
        "Enter Customer Index (0 to 8949):",
        min_value=0,
        max_value=len(df)-1,  # len(df)-1 = 8950-1 = 8949
        value=0,
        step=1
    )
 
    # df.iloc[index] → selects a row by its INTEGER position
    # iloc = integer location
    # (vs df.loc which selects by label/index name)
    # Returns a Series with column names as index and row values as values
    customer = df.iloc[customer_idx]
 
    # Extract cluster number from this customer's row
    # int() converts numpy int64 to Python int (needed for dict lookup)
    cluster_id = int(customer['Cluster'])
 
    # Display which segment this customer belongs to
    # f-string inserts variables directly into the string
    st.markdown(f"### {cluster_colors[cluster_id]} Customer #{customer_idx} belongs to:")
    st.markdown(f"## {cluster_names[cluster_id]}")
    st.markdown("---")
 
    # 6 metric cards showing this specific customer's actual values
    # (not averages — these are the real values for this one customer)
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
 
    # zip() pairs each column with its metric — same pattern as Tab 2
    for col, (label, value) in zip([c1,c2,c3,c4,c5,c6], lookup_metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

# To run: streamlit run dashboard/app.py