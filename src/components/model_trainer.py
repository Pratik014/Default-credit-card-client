import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test= (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1],

            )
            
            models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(criterion='gini',max_depth=7,n_estimators=6),
            "Support Vector machine":SVC()
            }
            params={
                "Decision Tree": {
                    'criterion':['gini'],
                    'splitter' : ['best', 'random'],
                    'max_features':['sqrt','log2']
                },
                
                "Logistic Regression":{
                    'penalty': ['l2',],
                
                },
                "Random Forest Classifier":{
                    'criterion':['gini'],
                 
                    'max_depth':[7],
                    'n_estimators': [6]
                    
                },
                "Support Vector machine":{
                    'gamma': ['scale','auto'],
                
            }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
        
        ## to get the best score from dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Retrieve the best model
            best_model = models[best_model_name]

            # Use string formatting to log the best model's information
            logging.info("Best model: %s", best_model_name)
            logging.info("Best model score: %f", best_model_score)

            logging.info("Best found model on both training and testing dataset:")
            logging.info(best_model)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Train and evaluate the best model
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Log the accuracy score
            logging.info("Accuracy: %f", accuracy)

            return {"accuracy": accuracy, "classification_report": report}
            
        except Exception as e:
            raise CustomException(e, sys)