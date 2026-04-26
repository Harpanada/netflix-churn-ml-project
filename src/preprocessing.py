import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Loading Dirty Dataset
df=pd.read_csv('data/raw/netflix_customer_churn.csv')

#Separate label and features
X= df.drop(columns='churned')
y= df['churned']

#Split data to test and train
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)

#Drop unused columns
drop_cols= ['customer_id']
X_train=X_train.drop(columns=drop_cols)
X_test=X_test.drop(columns=drop_cols)

#Remove outliers 
mask_train= X_train["avg_watch_time_per_day"] <= 24
mask_test=  X_test["avg_watch_time_per_day"]  <= 24

X_train=X_train[mask_train]
y_train=y_train[mask_train]

X_test=X_test[mask_test]
y_test=y_test[mask_test]

#Separate numeric and categorycal data
numeric_col= X_train.select_dtypes(include=["number"]).columns
categ_col= X_train.select_dtypes(include=['object', 'str']).columns


#Build preprocessing pipeline
num_pipeline=Pipeline([
    ('scaler',StandardScaler()),
])

cat_pipeline=Pipeline([
    ('encoder',OneHotEncoder()),
])

preprocessor=ColumnTransformer([
    ('num',num_pipeline,numeric_col),
    ('cat',cat_pipeline,categ_col )
])

#Returns data and preprocessors so they can be accessed in other modules.
def get_data_and_preprocessor():
    return X_train,X_test,y_train,y_test,preprocessor