import sys
from typing import Generator , List , Tuple
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV , train_test_split
from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utlis import MainUtlis

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join(artifact_folder , 'model_trainer')
    trained_model_path = os.path.join(model_trainer_dir , 'trained_model' , 'model.pkl')
    expected_accuracy = 0.45
    
    model_config_file_path = os.path.join('config' , 'model.yaml')
    
    
class VisibilityModel:
    def __init__(self , preprocessing_object  : ColumnTransformer , trained_model_object):
        self.preprocessing_object = preprocessing_object
        
        self.trained_model_object  = trained_model_object   
        
    def predict(self , X : pd.DataFrame)-> pd.DataFrame:
        logging.info("Entered predict method of srcTruckModel class")
        
        try:
            logging.info("Using trained to get prediction")
            
            transformed_features = self.preprocessing_object.transform(X)
            logging.info("Used the trained model to get prediction")
            
            return self.trained_model_object.predict(transformed_features)
        
        except Exception as e:
            raise CustomException(e , sys)
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    

class ModelTrainer():
    def __init__(self):
        
        self.model_trainer.config = ModelTrainerConfig()
        self.utils = MainUtlis()    
        
        self.models = {
            
            "GaussainNB" : GaussianNB(),
            "XGBClassifier" : XGBClassifier(objective = 'binary:logistic'),
            "LogisticRegression" :LogisticRegression()
        }    
    
    def evaluate_models(self , X_train , X_test , y_train , y_test , models):
        
        try:
            
            report = {}
            
            for i in range(len(list(models))):
                model = list(models.values())[i]
                
                model.fit(X_train , y_train)
                
                y_train_pred = model.predict(X_train)
                
                y_test_pred = model.predict(X_test)
                
                trained_model_score = accuracy_score(y_train , y_train_pred)
                
                test_model_score = accuracy_score(y_test , y_test_pred)
                
                report[list(models.keys()[i])] = test_model_score
                
            return report
        
        
        except Exception as e:
            raise CustomException(e , sys) from e
        
        
    def get_best_model(self , X_train :np.array,
                              y_train : np.array,
                              x_test : np.array,
                              y_test : np.array):
        
        try:
            
            model_report : dict = self.evaluate_models(
                X_train = X_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                models = self.models
                
            )
            
            print(model_report)
            
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[list
                                                        (model_report.values()).index(best_model_score)]
            
            
            best_model_object = best_model_object , best_model_score
                      
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    
    
    def fine_tune_best_model(self , best_model_object : object , best_model_name , X_train , y_train) -> object:     
        
        
        try:
            
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"][
                best_model_name]['best_param_grid']
            
            
            grid_search = GridSearchCV(best_model_object , parm_grid = model_param_grid , cv =5 , n_jobs= -1 , verbose=1)
            grid_search.fit(X_train , y_train)
            
            best_params = grid_search.best_params_
            
            print("best params are :"  , best_params)
            
            finetuned_model = best_model_object.set_params(**best_params)
            
            return finetuned_model
        
        
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    
    def initiate_model_trainer(self , X_train , y_train , X_test , y_test , preprocessor_path):
        
        try:
            
            logging.info(f"Splitting training and testing input and target feature")
            
            logging.info(f"Extracting model config file path") 
            
            preprocessor = self.utils.load_object(file_path= preprocessor_path)
            
            logging.info(f"Extracting model config file path") 
            
            
            model_report : dict = self.evaluate_models(X_train = X_train , y_train=y_train , X_test=X_test , y_test=y_test , models=self.models)
            
            best_model_score = max(sorted(model_report.values()))  
            
            best_model_name = list(model_report.keys())[list
                                                        (model_report.values()).index(best_model_score)]    
            
            best_model = self.models[best_model_name]
            
            best_model = self.fine_tune_best_model(
                best_model_name=best_model_name,
                best_model_object= best_model,
                X_train=X_train,
                y_train=y_train
            ) 
            
            
            best_model.fitX(X_train, y_train)
            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(X_test , y_pred)
            
            print(f"Best model name : {best_model_name} , and best model score : {best_model_score}")
            
            
            if best_model_score < 0.5:
                raise CustomException("No best model found with an accuracy greater than the threshold 0.6")
            
            logging.info("Best model found on training and testing dataset")
            
            custom_model = VisibilityModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model
            )
            
            logging.info(f"Saving model at path :{self.model_trainer_config.trained_model_path}")
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            
            self.utils.save_object(file_path=self.model_trainer_config.trained_model_path , obj=custom_model)
            
            self.utils.upload_file(from_filename=self.self.model_trainer_config.trained_model_path , to_filename="model.pkl",
                                   bucket_name=AWS_S3_BUCKET_NAME)
            
            
            return best_model_score
        
        
        except Exception as e:
            raise CustomException(e , sys) from e 