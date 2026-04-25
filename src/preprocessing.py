import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

df=pd.read_csv('data/raw/netflix_customer_churn.csv')

X= df.drop(columns='churned')
y= df['churned']

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)

drop_cols= ['customer_id']
X_train=X_train.drop(columns=drop_cols)
X_test=X_test.drop(columns=drop_cols)

numeric_col= X_train.select_dtypes(include=["int64","float64"]).columns
categ_col= X_train.select_dtypes(include=['object', 'str']).columns



 