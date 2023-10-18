import os
import sys

import dill


import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from src.logger import logging

def save_obj(file_path,obj):
    try :
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path,"wb") as file_obj :
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)






def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(model_name)
            parameters = param.get(model_name, {})
            
            if parameters:
                grid_search = GridSearchCV(model, parameters, cv=3)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model

            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate classification metrics
            f1 = f1_score(y_test, y_test_pred, average='weighted')
#            accuracy = accuracy_score(y_test, y_test_pred)
#            precision = precision_score(y_test, y_test_pred, average='weighted')
#            recall = recall_score(y_test, y_test_pred, average='weighted')
#            conf_matrix = confusion_matrix(y_test, y_test_pred)
#            class_report = classification_report(y_test, y_test_pred)

            report[model_name] = {
                "F1 Score": f1,
#                "Accuracy": accuracy,
#                "Precision": precision,
#                "Recall": recall,
#                "Confusion Matrix": conf_matrix,
#                "Classification Report": class_report
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)


