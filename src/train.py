from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from preprocessing import get_data_and_preprocessor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib


#Get data and preprocessor
X_train, X_test, y_train, y_test, preprocessor = get_data_and_preprocessor()

#Build Models
models={
    "base_log_reg":Pipeline([
        ('prep',preprocessor),
        ('clf',LogisticRegression())
    ]),
    "Dec_tree":Pipeline([
        ('prep',preprocessor),
        ('Dec_tre',DecisionTreeClassifier())

    ]),
    "KNN":Pipeline([
        ('prep',preprocessor),
        ('Dec_tre',KNeighborsClassifier())

    ]),
    "Gaussian_NB":Pipeline([
        ('prep',preprocessor),
        ('Dec_tre',GaussianNB())

    ]),

}

#Train and test models
result=[]
best_acc  = 0
best_prec   = 0  
best_recl = 0
best_f1= 0
best_model = None
best_name  = None

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    accuration= accuracy_score(y_test,y_pred)
    precision=  precision_score(y_test,y_pred)
    recall=     recall_score(y_test,y_pred)
    f1=         f1_score(y_test,y_pred)

    result.append({
        'Model'       : name,
        'Accuration'  : accuration,
        'Precision'   : precision,
        'Recall'      : recall,
        'f1_score'    : f1,
        'Score_Train' : model.score(X_train, y_train),
        'Score_Test'  : model.score(X_test, y_test)
    })

    if accuration > best_acc or precision > best_prec or recall > best_recl or f1 > best_f1:
        best_acc        = accuration
        best_prec       = precision
        best_recl       = recall
        best_f1         = f1
        best_model      = model
        best_name       = name

result_df=pd.DataFrame(result)
result_df=result_df.sort_values('f1_score',ascending=False)

print(result_df)
print(f"\n🏆 Best Model : {best_name}")
