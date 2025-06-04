'''the goal here is to autpmatically classify each cols in a pandas DF into one
of four types:
1. Numerical: continuous or discrete numeric data
2.categorical: discrete values with limited unique options
3.text: free form string/ text data
4. datetime: date/time valiues
'''


def detect_feature_types(df):
    """
    classify each col into numerical, categorical, text and datetime"""

    feature_types={}
    for col in df.columns:
        col_data=df[col]
        unique_ratio= len(col_data.dropna().unique()) / len(col_data.dropna())
        """calculate the unique ration:
        if all values are unique: unique ratop == 1
        else unique rations will be less
        """
        if col_data.dtype == 'O' or unique_ratio > 0.5:
            """'O'means object datatype, meaningg if cols contains strings or mixrd types,
            or has a high ratio of unique values, eg:[1,'two',3]"""
            feature_types[col]= 'text'
        elif col_data.apply (lambda x: isinstance(x, str)).any():
            feature_types[col]= 'text'
        elif col_data.dtype in ['datetime64[ns]','datetime']:
            feature_types[col]= 'datetime'
        else:
            """if its numeric but has few unique values, it is considered categorical
            rlse it is labelled as numerical"""
            if unique_ratio < 0.5:
                feature_types[col]= 'categorical'
            else:
                feature_types[col]= 'numerical'
        
    return feature_types
