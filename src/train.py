from sklearn.linear_model import LogisticRegression
from preprocessing import get_data_and_preprocessor
from sklearn.pipeline import Pipeline

#Get data and preprocessor
X_train, X_test, y_train, y_test, preprocessor = get_data_and_preprocessor()

#Build Models
models={
    "base_logreg":Pipeline([
        ('prep',preprocessor),
        ('clf',LogisticRegression())
    ]),
}