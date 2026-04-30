from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from preprocessing import get_data_and_preprocessor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# ─────────────────────────────────
# GET DATA AND PREPROCESSOR
# ─────────────────────────────────
X_train, X_test, y_train, y_test, preprocessor = get_data_and_preprocessor()

# ─────────────────────────────────
# BUILD MODELS WITH PIPELINE 
# ─────────────────────────────────
models={
    "base_log_reg":Pipeline([
        ('prep',preprocessor),
        ('model',LogisticRegression())
    ]),
    "Dec_tree":Pipeline([
        ('prep',preprocessor),
        ('model',DecisionTreeClassifier())

    ]),
    "KNN":Pipeline([
        ('prep',preprocessor),
        ('model',KNeighborsClassifier())

    ]),
    "Gaussian_NB":Pipeline([
        ('prep',preprocessor),
        ('model',GaussianNB())

    ]),

}

# ─────────────────────────────────
# TRAIN AND TEST MODEL
# ─────────────────────────────────
result=[]
best_acc  = 0
best_prec   = 0  
best_recl = 0
best_f1= 0
best_model = None
best_name  = None

for name,model in models.items():
    #TRAIN
    model.fit(X_train,y_train)
    #TEST
    y_pred=model.predict(X_test)

    #EVALUATION METRIC
    accuration= accuracy_score(y_test,y_pred)
    precision=  precision_score(y_test,y_pred)
    recall=     recall_score(y_test,y_pred)
    f1=         f1_score(y_test,y_pred)

    #SAVE MODELS PERFORMANCE INFORMATION
    result.append({
        'Model'       : name,
        'Accuration'  : accuration,
        'Precision'   : precision,
        'Recall'      : recall,
        'f1_score'    : f1,
        'Score_Train' : model.score(X_train, y_train),
        'Score_Test'  : model.score(X_test, y_test)
    })

    #COMPARE MODELS WITH ONE ANOTHER TO CHOOSE THE BEST ONE
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

best_name=f'Trained_{best_name}.pkl'
best_mod_dir= f'./models/{best_name}'

# ─────────────────────────────────
# SAVE THE BEST MODEL
# ─────────────────────────────────
joblib.dump(best_model,best_mod_dir)
print(f"\n✅ Best Model Saved as : {best_name} ")