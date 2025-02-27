import os
import sys
import pandas as pd
import sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utlis import MainUtlis
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(artifact_folder , 'data_transformation')
    transformed_train_file_path = os.path.join(data_transformation_dir , 'train.npy')
    transformed_test_file_path = os.path.join(data_transformation_dir , 'test.npy')
    transformed_object_file_path = os.path.join(data_transformation_dir , 'prepocessing.pkl')
    
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtlis()
        
    
    @staticmethod
    def get_merged_batch_data(valid_data_dir : str) -> pd.DataFrame:
        
        """ Description : This method will read all the validate raw data from valid_data_dir and return a panda DataFrame 
        """
        try:
            raw_files = os.listdir(valid_data_dir)
            csv_data = []
        
            for filename in raw_files:
                data = pd.read_csv(os.path.join(valid_data_dir , filename))
                csv_data.append(data)

                merged_data = pd.concat(csv_data)
        
                return merged_data
            
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    
    
    def initiate_data_transformation(self):
        """ Description : This method initiate the transformation component of pipeline
        """
        
        logging.info("Entered initiate_data_transformation method of Data_Transformation_class")  
        
        
        try:
            dataframe = self.get_merged_batch_data(valid_data_dir=self.valid_data_dir)
            dataframe = self.utils.remove_unwanted_spaces(dataframe)
            
            dataframe.replace({"?": np.nan} , inplace= True)  
            
            
            x = dataframe.drop(columns=Target_COLUMN)
            y = np.where(dataframe[Target_COLUMN] == -1 , 0 ,1 )
            
            sampler = RandomOverSampler()
            
            x_sampled  , y_sampled = sampler.fit_resample(x , y)
            
            X_train , X_test , y_train , y_test = train_test_split(x_sampled , y_sampled , test_size=0.2)
            
            preprocessor = SimpleImputer(strategy='most_frequent')
            
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.fit(X_test)
            
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path) , exist_ok=True)
            
            self.utils.save_object(file_path=preprocessor_path  , obj=preprocessor)
            
            return X_train_scaled , y_train , X_test_scaled , y_test , preprocessor_path
            
        except Exception as e:
            raise CustomException(e , sys) from e  
    
            

    