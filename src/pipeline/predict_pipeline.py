import shutil
import numpy as np
import sys , os
import pandas as pd
import flask

from src.logger import logging
from src.constants import *
from src.exception import CustomException
from src.utils.main_utlis import MainUtlis
from flask import request

from dataclasses import dataclass


@dataclass
class PredictionFileDetail:
    prediction_output_dirname : str = "prediction"
    prediction_file_name = "predicted_file.csv"
    prediction_file_path :str = os.path.join(prediction_output_dirname , prediction_file_name)
    
    

class PredictionPipeline:
    def __init__(self , request: request):
        
        self.request = request
        self.utils = MainUtlis()
        self.prediction_file_details = PredictionFileDetail()
        
        
    def save_input_file(self) -> str:
        
        """ Description : This method saves the input file to the prediction artifact directory
        """        
        try:
            
            pred_file_input_dir = "prediction_artifact"
            os.makedirs(pred_file_input_dir , exist_ok= True)
            
            input_csv_file = self.request.files["file"]
            
            pred_file_path = os.path.join(pred_file_input_dir , input_csv_file.filename)
            
            input_csv_file.save(pred_file_path)
            
            return pred_file_path
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def predict(self , features):
        try:
            model_path  = self.utils.download_model(
            bucket_name=AWS_S3_BUCKET_NAME,  # Assuming this should be the S3 bucket
            bucket_file_name='model.pkl',  # The actual model file in the bucket
            desti_file_name="model.pkl"
       )
 
               
            
            
            model = self.utils.load_object(file_path=model_path)
            
            preds = model.predict(features)
            
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    def get_predicted_dataframe(self , input_dataframe_path : pd.DataFrame):
        
        """Description : This method returns tha dataframe with new columns containing prediction 
        """       
        
        try:
            
            prediction_column_name : str = Target_COLUMN
            input_dataframe : pd.DataFrame = pd.read_csv(input_dataframe_path)
            predictions = self.predict(input_dataframe)
            
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            
            target_column_mapping = {0 : 'phishing' , 1 : 'safe'}
            
            
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedir(self.prediction_file_details.prediction_output_dirname , exist_ok = True)
            
            input_dataframe.to_csv(self.prediction_file_details.prediction_file_path , index= False)
            
            logging.info("Prediction Compeleted")
            
            
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_file()
            self.get_predicted_dataframe(input_csv_path) 
            
            return self.prediction_file_details
        
        except Exception as e:
            raise CustomException(e , sys) from e 
        
             