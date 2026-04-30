import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ─────────────────────────────────
# LOADING DIRTY DATASET
# ─────────────────────────────────
df=pd.read_csv('data/raw/netflix_customer_churn.csv')

# ─────────────────────────────────
# SEPARATE LABEL AND FEATURES
# ─────────────────────────────────
X= df.drop(columns='churned')
y= df['churned']

# ─────────────────────────────────
# SPLIT DATA TO TRAIN SET AND TEST SET
# ─────────────────────────────────

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)

# ─────────────────────────────────
# DROP UNUSED COLUMN
# ─────────────────────────────────
drop_cols= ['customer_id']
X_train=X_train.drop(columns=drop_cols)
X_test=X_test.drop(columns=drop_cols)

# ─────────────────────────────────
# REMOVE OUTLIER
# ─────────────────────────────────
mask_train= X_train["avg_watch_time_per_day"] <= 24
mask_test=  X_test["avg_watch_time_per_day"]  <= 24

X_train=X_train[mask_train]
y_train=y_train[mask_train]

X_test=X_test[mask_test]
y_test=y_test[mask_test]

# ─────────────────────────────────
# SEPARATE NUMERIC AND CATEGORYCAL COLUMN
# ─────────────────────────────────
numeric_col= X_train.select_dtypes(include=["number"]).columns
categ_col= X_train.select_dtypes(include=['object', 'str']).columns


# ─────────────────────────────────
# BUILD PREPROCESSING PIPELINE
# ─────────────────────────────────

num_pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('imputer', SimpleImputer(strategy= "median"))
])

cat_pipeline=Pipeline([
    ('encoder',OneHotEncoder()),
    ('imputer',SimpleImputer(strategy="most_frequent"))
])

preprocessor=ColumnTransformer([
    ('num',num_pipeline,numeric_col),
    ('cat',cat_pipeline,categ_col )
])

# ─────────────────────────────────
# RETURN CLEAN DATA AND PREPROCESSOR 
# ─────────────────────────────────
def get_data_and_preprocessor():
    return X_train,X_test,y_train,y_test,preprocessor