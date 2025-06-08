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
import os
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
        df = load_data(uploaded_file.name)
        st.success("Dataset loaded successfully!") 

        analyzer= EDAAnalyzer(df)
        plotter= PlotGenerator(df)

        #Basoc Stats:
        st.subheader("Dataset Summary")
        stats = analyzer.get_basic_stats()
        st.write(f"Rows: {stats['shape'][0]}, Columns: {stats['shape'][1]}")
        st.write(f"Missing Values: {stats['missing_values']}")
        st.write(f"Duplicate Rows: {stats['n_duplicates']}")


        #Feature summary
        st.subheader("Feature Summary")
        feature_summary= pd.DataFrame(analyzer.feature_summary())
        st.dataframe(feature_summary)

        #select cols for plots
        selected_col= st.selectbox("Select a column to visualize", options= df.columns.tolist())

        #Plot options:
        plot_type= st.radio("Select plot type", [
            "Histogram",
            "Boxplot",
            "Bar Chart",
            "Correlation Heatmap",
            "PCA Plot",
            "t-SNE Plot"
        ])

        if plot_type == "Histogram":
            plotter.plot_histogram(selected_col)
        elif plot_type == "Boxplot":
            plotter.plot_boxplot(selected_col)
        elif plot_type == "Bar Chart":
            if df[selected_col].dtype == 'O':
                plotter.plot_bar_chart(selected_col)
            else:
                st.warning("Bar chart is only supported for categorical features.")
        elif plot_type == 'Correlation Heatmap':
            plotter.plot_correlation_heatmap()
        elif plot_type == "PCA Plot":
            target_col =  st.selectbox("Select target column for coloring", options= df.columns.tolist(), key='pca_target')
            plotter.plot_pca(target_col=target_col)
        elif plot_type == "t-SNE Plot":
            target_col= st.selectbox("Select target column for coloring", options=df.columns.tolist(), key= "tsne_target")
            plotter.plot_tsne(target_col= target_col)
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.info("Please upload a dataset to begin EDA")


#error in finding the backend module