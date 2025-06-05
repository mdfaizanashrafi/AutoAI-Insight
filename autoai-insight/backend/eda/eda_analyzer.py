import pandas as pd
import numpy as np

class EDAAnalyzer:
    def __init__(self,df):
        self.df = df
    

    def get_basic_stats(self):
        """returns basic statistics of the daataset"""
        stats={
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'n_duplicates': self.df.duplicated().sum()
        }
        return stats
    

    def feature_summary(self):
        """summarize each fetures stats
        loops through each cols and return a list of dictionaries containing
        detailed feature stats"""

        summary=[]
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            missing_count = self.df[col].isnull().sum()
            if 'int' in dtype or 'float' in dtype:
                mean = round(self.df[col].mean(),2)
                std = round(self.df[col].std(),2)
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                summary.append({
                    'Feature': col,
                    'Type': 'Numerical',
                    'Unique Values': unique_count,
                    'Missing': missing_count,
                    'Mean': mean,
                    'Std': std,
                    'Min': min_val,
                    'Max': max_val
                })

            else:
                top_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                freq = self.df[col].value_counts().iloc[0] if not self.df[col].empty else 0
                summary.append({
                    'Feature': col,
                    'Type': 'Categorical',
                    'Unique Values': unique_count,
                    'Missing': missing_count,
                    'Top Value': top_value,
                    'Frequency': freq
                })
        
        return summary
    


    def detect_outliers(self, col, method='iqr'):
        """detect outliers in a numerical colsusing IQR: InterQuartile range
        IQR is a measure of statistical dipersion: how spread out the middle 50%
        pf our data is.
        Q1= the value below which 25% of the data falls
        Q3= the value below which 75% of data falls
        IQR= Q3-Q1
        """
        q1 = self.df[col].quantile(0.25)
        q3= self.df[col].quantile(0.75)
        iqr=q3-q1
        lower_bound= q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr

        return self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]