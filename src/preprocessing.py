import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

#Loading Dirty Dataset
df=pd.read_csv('data/raw/netflix_customer_churn.csv')

#Separate label and features
X= df.drop(columns='churned')
y= df['churned']

#Split data to test and train
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)



