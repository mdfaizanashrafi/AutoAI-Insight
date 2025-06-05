#Streamlit based interactive EDA webapp that helps users perform quick explanatory
#data anakysis on their datasets:
"""
🔍 1. Data Understanding
See number of rows/columns
Understand missing values and duplicates
Get detailed feature summaries (mean, std, unique values, etc.)
📊 2. Interactive Data Visualization
Histograms for numeric columns
Boxplots to detect outliers
Bar charts for categorical variables
Correlation heatmap for numerical columns
Dimensionality reduction plots (PCA and t-SNE)
🎨 3. User-Friendly Interface
Drag-and-drop file upload
Interactive dropdowns and widgets
Real-time visualizations
🛠️ 4. Modular Architecture
Uses load_data, EDAAnalyzer, and PlotGenerator from separate modules
Easy to extend or plug into larger ML pipelines
"""

import streamlit as st
import pandas as pd
from backend.data_pipeline.data_loader import load_data
from backend.eda.eda_analyzer import EDAAnalyzer
from backend.eda.plot_generator import PlotGenerator

st.set_page_config(page_title="AutoAI Insight - EDA", layout="wide")
st.title("📊 AutoAI Insight: EDA Module")

uploaded_file= st.file_uploader("Upload your dataset (.csv, .parquet, .json)", type=["csv","parquet","json"])

if uploaded_file is not None:
    try:
        

