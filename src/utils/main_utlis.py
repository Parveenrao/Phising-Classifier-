import sys
from typing import Dict , Tuple

import os
import pandas  as pd
import numpy as np
import pickle
import yaml
import boto3


from src.constants import *
from src.logger import logging
from src.exception import CustomException

class MainUtlis:
    def __init__(self) -> None:
        pass
    
    def read_yaml_file(self , filename:str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            raise CustomException(e , sys) from e
        
     
    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config" , "training_schema.yaml"))
            
            return schema_config
        
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    
    @staticmethod
    def save_object(file_path : str , obj: object) -> None:
        logging.info("Entered the save_object method of MainUtlis class")
        
        try:
            with open(file_path , "wb") as file_obj:
                pickle.dump(obj , file_obj)
            
            logging.info("Exicted the save_object method of MainUtlis class")
                      
        except Exception as e:
            raise CustomException(e , sys) from e
        
    
    @staticmethod
    def load_object(file_path:str) -> object:
        logging.info("Entered the load_object method of MainUtlis class")
        
        try:
            with open(file_path , "rb") as file_obj:
                obj  = pickle.load(file_obj)
                
            logging.info("Exited the load_object method of MainUtlis class")
            
            return obj
        
        
        except Exception as e:
            raise CustomException(e ,sys) from e
        
    
    @staticmethod
    def upload_file(from_filename , to_filename , bucket_name):
        try:
            s3_resouce = boto3.resource("s3")
            s3.resource.meta.client.upload_file(from_filename , bucket_name , to_filename)      
         
        except Exception as e:
            raise CustomException(e ,sys)

    @staticmethod
    def download_model(bucket_name , bucket_file_name , desti_file_name):
        try:
            s3_client = boto3.client("s3")
            s3_client.download_file(bucket_name , bucket_file_name , desti_file_name)
            
            return desti_file_name
        
        except Exception as e:
            raise CustomException(e , sys) from e
        
        
    @staticmethod
    def remove_unwanted_spaces(data: pd.DataFrame) -> pd.DataFrame:
        
        
                
        """ Description : This method remove unwanted spaces from a panda dataframe
                 
        """  
                                                      
        try:
            df_without_space = data.apply(
                lambda x: x.str.strip() if x.dtype == "object" else x )
            
            logging.info("Unwanted Spaces removal Successful")
            
            return df_without_space
        
        except Exception as e:
            raise CustomException(e , sys) from e
        
        
    @staticmethod
    def identify_features_types(dataframe:pd.DataFrame):
        data_types = dataframe.dtypes 
        
        categorical_features = []
        continous_features =   []
        discrete_features =    []
        
        
        for column , dtype in dict(data_types).items():
            unique_values = dataframe[column].nunique()
            
            
            if dtype == 'object' or unique_values < 10:
                categorical_features.append(column)
            
            elif dtype in [np.int64 , np.float64]:
                if unique_values > 20:
                    continous_features.append(column)
                 
                else:
                    discrete_features.append(column)
            
            else:   # Handle Other data type is possible
                pass               
                                                          
                                                          
        return continous_features , continous_features , discrete_features                                                  