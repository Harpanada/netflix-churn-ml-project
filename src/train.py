from sklearn.linear_model import LogisticRegression
from preprocessing import get_data_and_preprocessor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

#Get data and preprocessor
X_train, X_test, y_train, y_test, preprocessor = get_data_and_preprocessor()

#Build Models
models={
    "base_log_reg":Pipeline([
        ('prep',preprocessor),
        ('clf',LogisticRegression())
    ]),
}

#Train and test models
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"""Model: {name}
          \nScore: {model.score(X_test,y_test):,.2f}
          \nClassification Report:
          \n{classification_report(y_test,y_pred)}
          \nConfusion Matrix:
          \n{confusion_matrix(y_test,y_pred)}
          \nCross Val:
          \n{cross_val_score(model, X_train, y_train, cv=5)}
    """)
