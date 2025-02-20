import sys 
import os 
from typing import Optional , List
from database_connect import mongo_operation as mongo
from pymongo import MongoClient
import numpy as np
import pandas as pd
from src.constants import * 
from src.configuration.mogo_db_connection import MongoDBClient
from src.exception import CustomException


class PhisingData:
    def __init__(self , database_name : str):
        
        try:
            self.database_name = database_name
            self.mongo_url = os.getenv("client_url")
        
        except Exception as e:
            raise CustomException(e , sys) from e
        
    
    def collection_name(self) -> List:
        
        mongo_db_client = MongoClient(self.client_url)
        collection_name  = mongo_db_client[self.database_name].list_collection_names()
        return collection_name
    
    
    
    def get_collection_data(self , collection_name : str) -> pd.DataFrame:
        
        mongo_connection = mongo(
            client_url = self.client_url,
            database_name = self.database_name,
            collection_name = collection_name
        )
        
        df  = mongo_connection.find()
        
        if "_id" in df.columns.to_list():
            df = df.drop(columns = ["_id"])    
         
        df = df.replace({"na" : np.nan})    
        return df
    
    
    
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        
        try:
            collections = self.get_collection_names()
            
            for collection_name in collections:
                df = self.get_collection_data(collection_name=collection_name)
                
                yield collection_name , df
                
        except Exception as e:
            raise CustomException(e ,sys) from e        
            