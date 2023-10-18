import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,save_obj


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "classification_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
#               "Gradient Boosting": GradientBoostingClassifier(),
#                "Logistic Regression": LogisticRegression(),
#                "K-Nearest Neighbors": KNeighborsClassifier(),
#                "XGBoost": XGBClassifier(),
#                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                },
                "Random Forest": {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
#                "Gradient Boosting": {
#                   'loss': ['deviance', 'exponential'],
#                    'learning_rate': [0.01, 0.05, 0.1],
#                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
#                    'n_estimators': [8, 16, 32, 64, 128, 256]
#                },
#                "Logistic Regression": {},
#                "K-Nearest Neighbors": {
#                    'n_neighbors': [3, 5, 7, 9],
#                    'weights': ['uniform', 'distance'],
#                },
#                "XGBoost": {
#                    'learning_rate': [0.01, 0.05, 0.1],
#                    'n_estimators': [8, 16, 32, 64, 128, 256]
#                },
#                "AdaBoost": {
#                    'learning_rate': [0.01, 0.05, 0.1],
#                    'n_estimators': [8, 16, 32, 64, 128, 256]
#                }  

                
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Get the best model name based on F1 score
            best_model_name = max(model_report, key=lambda model: model_report[model]['F1 Score'])            
            best_model = models[best_model_name]

            if model_report[best_model_name]["F1 Score"] < 0.6:  # Adjust this threshold as needed
                 raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name}")

            # Save the best model
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

 

            return best_model, model_report
        
        

        except Exception as e:
            raise CustomException(e, sys)
