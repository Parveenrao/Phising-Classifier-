import os 
import sys 
import numpy as np
import pandas as pd
from pymongo import MonngoClient
from zipfile import Path
from src.constants import *
from src.exception import CustomException
from src.logger import logging

from src.data_access.phising_data import PhisingData
from src.utils.main_utlis import MainUtlis
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_ingestion_dir : str = os.path.join(artifact_folder , "data_ingestion")
    

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtlis()
        
    def export_data_into_raw_data_dir(self) -> pd.DataFrame:
        
        """Description : This method export data from mongo_db and store into artifacts
           Output : Output is returned as pd.DataFrame
        """      
        try:
            logging.info("Exporting data from mongodb")
            raw_batch_files_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_batch_files_path , exist_ok = True)
            
            
            incoming_data = PhisingData(database_name=MONGO_DATABASE_NAME)
            
            logging.info(f"Saving exported data into feature store file path : {raw_batch_files_path}")
            for collection_name , dataset in incoming_data.export_collection_as_dataframe:
                logging.info(f"Shape of {collection_name} : {dataset.shape}")
                feature_store_file_path = os.path.join(raw_batch_files_path , collection_name + '.csv')
                print(f"feature_store_file_path {feature_store_file_path}")
            
                dataset.to_csv(feature_store_file_path , index = False)
            
            
        except Exception as e:
            raise CustomException(e , sys) from e
        
        
        
    def initate_data_ingestion(self) -> Path:
        
        """Description : This method initate the data ingestion component of training pipeline
        """
        logging.info("Entered the initate_data_ingestion of Data Ingestion class")    
        
        try:
            self.export_data_into_raw_data_dir()
            
            logging.info("Got the data from mongodb")
            
            logging.info("Exited the initate_data_ingestion method of Data Ingestion class")
            
            
            return self.data_ingestion_config.data_ingestion_dir
        
        except Exception as e:
            raise CustomException(e , sys) from e