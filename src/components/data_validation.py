import os
import numpy as np
import pandas as pd
import sys
import re
import json
import shutil


from env.Lib.path.pathlib import Path
from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utlis import MainUtlis
from dataclasses import dataclass

LENGTH_OF_DATE_STAMP_IN_FILE = 8
LENGTH_OF_TIME_STAMP_IN_FILE = 6

NO_OF_COLUMNS = 31

@dataclass
class DataValidationConfig:
    data_validation_dir : str = os.path.join(artifact_folder , 'data_validation')
    valid_data_dir : str = os.path.join(data_validation_dir , 'validated')
    invalid_data_dir :str = os.path.join(data_validation_dir , 'invalid')
    
    schema_config_file_path :str = os.path.join('config' , 'training_schema.json')
    
class DataValidation:
    def __init__(self , raw_data_store_dir : Path):
        self.raw_data_store_dir = raw_data_store_dir
        self.data_validation_config = DataValidationConfig()
        self.utils = MainUtlis()
       
    def ValueFromSchema(self):
        
        """Description : This method extract all the relevant information from pre_defined "Schema" 
            
           OUtput : LengthofDateStampFile , LengthOfTimeStampFile  """
                
        try:
            
            with open(self.data_validation_config.schema_config_file_path , 'r')  as f:
                dic = json.load(f)
                f.close()       
               
            LengthOfDateStampInFile = dic[' LengthOfTimeStampInFile']
            LengthOfTimeStampInFile  = dic['LengthOfTimeStampInFile']    
            
            column_name = dic['ColName']
            no_of_columns = dic['NumberOfColumns']
            
            
            return  LengthOfDateStampInFile , LengthOfTimeStampInFile , column_name , no_of_columns
        
        except Exception as e:
            raise CustomException(e , sys) from e  
    
    
    def validate_file_name(self , file_path: str , length_of_date_stamp : int , length_of_time_stamp : int) -> bool:
        
        """ Description : This method validate the file name for a raw file  
        """  
        
        try:
            file_name = os.path.basename(file_path)
            regex = "['phishing] + ['\_''] +[\d_] + [\d] + \.csv"
            if re.match(file_path , regex):
                splitAtDot = re.split('.csv', file_name)
                splitAtDot = (re.split('-' , splitAtDot[0]))  
                filename_validation_status = len(splitAtDot[1]) == length_of_date_stamp and len(splitAtDot[2] == length_of_time_stamp)
            
            else:
                filename_validation_status = False
            
            return filename_validation_status        
        
        except Exception as  e:
            raise CustomException(e , sys) from e 
        
    
    def validate_no_of_columns(self , file_path : str , schema_no_of_columns : int) -> bool:
        
        """Description : This method validate the number of columns for a particular raw file
        
        """    
        try: 
           dataframe = pd.read_csv(file_path)
           column_length_validation_status = len(dataframe.columns) == schema_no_of_columns 
           
           return column_length_validation_status
       
       
        except Exception as e:
           raise CustomException(e , sys) from e
    
    
    def validating_missing_values_in_whole_column(self , file_path : str) -> bool:
        
        """Description : This method validate if there is any column in dcsv file which has all values as null
        """ 
        
        try: 
            dataframe = pd.read_csv(file_path)
            no_of_columns_with_whole_null_values = 0
            for columns in dataframe:
                if(len(dataframe[columns]) - dataframe[columns].count()) == len(dataframe[columns]):
                    no_of_columns_with_whole_null_values += 1
                    
            if no_of_columns_with_whole_null_values == 0:
                missing_value_validation_status = True
            
            else:
                missing_value_validation_status = False  
                
            return missing_value_validation_status               
        
        except Exception as e:
            raise CustomException(e , sys) from e   
        
        
    def  get_raw_batch_files_path(self) -> list:
            
        """ Description : This method return all the raw files dir path in List
        """     
        try:
            raw_batch_files_names =  os.listdir(self.raw_data_store_dir)
            raw_batch_files_paths = [os.path.join(self.raw_data_store_dir , raw_batch_file_name) for raw_batch_file_name in raw_batch_files_names]
            return raw_batch_files_paths
            
        except Exception as e:
           raise CustomException(e , sys) from e
       
       
    def move_raw_files_to_validation_dir(self , src_path : str , dest_path : str):
        
        """Description : This method move validated raw_files to validation directory
        """  
        try:
            os.makedirs(dest_path , exist_ok= True)
            if os.path.basename(src_path not in os.listdir(dest_path)):
                shutil.move(src_path , dest_path) 
        
        except Exception as e:
            raise CustomException(e , sys) from e
        
    
    def validate_raw_files(self) -> bool:
        
        """Description : This method validate the raw files for training 
        """    
        try:
            raw_batch_files_paths = self.get_raw_batch_files_path()
            length_of_date_stamp , length_of_time_stamp , column_name , no_of_columns = self.ValueFromSchema()
            
            validate_files = 0
            
            for raw_file_path  in raw_batch_files_paths: 
                file_name_validation_status = self.validate_file_name(raw_file_path ,
                                                                      length_of_date_stamp= length_of_date_stamp,
                                                                      length_of_time_stamp=length_of_time_stamp)
                
                
                column_length_validation_status = self.validate_no_of_columns(raw_file_path , schema_no_of_column=no_of_columns)
                
                missing_value_validation_status = self.validating_missing_values_in_whole_column(raw_file_path)
                
                
                if(file_name_validation_status , column_length_validation_status  , missing_value_validation_status):
                    validate_files += 1
                    
                    self.move_raw_files_to_validation_dir(raw_file_path ,self.data_validation_config.valid_data_dir)
                    
                else:
                    self.move_raw_files_to_validation_dir(raw_file_path , self.data_validation_config.invalid_data_dir)    
                
            validation_status = validate_files > 0
            
            return validation_status    
       
        except Exception as e:
            raise CustomException(e , sys) from e 
        
    
    def initiate_data_validation(self):
        
        """Description : Initiate the data validation component of the piepline
        """    
        
        logging.info("Entered initiate data validation  method of Data_validation class")
        
        try:
            logging.info("Initiated the validation of the dataset")
            validation_status  = self.validate_raw_files()
            
            if validation_status:
                valid_data_dir = self.data_validation_config.valid_data_dir
                return valid_data_dir
            
            else:
                raise Exception("No data could be validated . Pipeline Stopped")
            
            
        except Exception as e:
            raise CustomException(e , sys) from e    