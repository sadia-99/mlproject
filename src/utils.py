import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

'''save_object(file_path, obj)
  ├─ 1. Récupérer dossier du chemin
   ├─ 2. Créer dossier si inexistant
   ├─ 3. Ouvrir fichier en écriture binaire
   ├─ 4. Sérialiser l'objet (dill)
   └─ 5. Gérer erreurs avec CustomException
'''

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            #Hyperparameter tunning
            #debut
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #fin
            #model.fit(X_train, y_train) 
            #prediction
            y_train_pred = model.predict(X_train) #prediction for train 
            y_test_pred = model.predict(X_test) #prediction for test

            #Evaluation des models si y  pas d'overfitting des models 
            train_model_score = r2_score(y_train, y_train_pred) #Evaluation des préditions train 
            test_model_score = r2_score(y_test, y_test_pred) #Evaluation des préditions test

            #print(gs.best_params_)
            #print(f"Train model score : {train_model_score}")
            #print(f"test model score: {test_model_score}")
            report [list(models.keys())[i]] = (test_model_score, gs.best_params_)
        return report
    except Exception as e:
        raise CustomException (e,sys)
    
#function for predcition pipeline
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        