#plots for EDA(explanatoruy Data analysis)/ model interpretations

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA  #dimenstionality reduction using Principal Componenet Analyusis
from sklearn.manifold import TSNE #non linear dimensionality rediction for visualization

class PlotGenerator:
    def __init__(self,df):
        self.df = df

    
    def plot_histogram(self, col, bins=20):
        """visualizes the distribution of a numerical cols
        inclides KDE(Kernel Density Estimate)"""
        plt.figure(figsize=(8,4))
        sns.histplot(self.df[col], kde=True, bins=bins)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    
    def plot_boxplot(self,col):
        """shows summary stats and outliers in a numerical cols
        based on IQR"""
        plt.figure(figsize=(6,4))
        sns.boxplot(x=self.df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

    
    def plot_bar_chart(self, col):
        """display freq of categories in a categorical variable
        uses y= col --> horizponatal bar chart"""
        plt.figure(figsize=(8,4))
        sns.countplot(data=self.df, y=col)
        plt.title(f"Barchart of {col}")
        plt.tight_layout()
        plt.show()

    
    def plot_correlation_heatmap(self):
        """visulizes correlation between numerical colimns
        helps detect multicollinearity"""
        numeric_df= self.df.select_dtypes(include= np.number)
        corr = numeric_df.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    
    def plot_pca(self, target_col=None):
        """redices high dimentionnal data into 2D for visualization
        optionally colors poinnts based on target(classification)"""
        numeric_df = self.df.select_dtypes(include=np.number).dropna()
        pca= PCA(n_components=2)
        components= pca.fit_transform(numeric_df.drop(columns= [target_col], errors='ignore'))
        explained_variance= pca.explained_variance_ratio_
        df_pca = pd.DataFrame(data= components, columns=['PC1','PC2'])
        if target_col:
            df_pca['Target']= self.df.loc[numeric_df.index, target_col]
            fig = px.scatter(df_pca, x='PC1',y='PC2',color='Target', title=f"PCA (Explained Variance: {explained_variance})")
        else:
            fig=px.scatter(df_pca,x='PC1',y='PC2',title=f"PCA (Explained Variance: {explained_variance})")
        fig.show()


    def plot_tsne(self, target_col=None, perplexity= 30):
        """better than PCA for non linear structures
        commonly used for clustering visualization"""
        numeric_df = self.df.select_dtypes(include=np.number).dropna()
        tsne= TSNE(n_components=2, perplexity=perplexity,random_state=42)
        components= tsne.fit_transform(numeric_df.drop(columns=[target_col], errors='ignore'))
        df_tsne= pd.DataFrame(data=components, columns=['Dim1', 'Dim2'])

        if target_col:
            df_tsne['Target']= self.df.loc[numeric_df.index, target_col]
            fig = px.scatter(df_tsne, x='Dim1', y='Dim2', color='Target', title='t-SNE Plot')
        else:
            fig=px.scatter(df_tsne, x='Dim1',y='Dim2', title='t-SNE Plot')
        
        fig.show()