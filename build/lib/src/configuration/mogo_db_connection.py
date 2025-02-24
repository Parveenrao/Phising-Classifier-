import os
import sys

import certifi
import pymongo

from src.constants import *
from src.exception import CustomException

ca = certifi.where()

class MongoDBClient:
    client = None
        
    def __init(self , database_name = MONGO_DATABASE_NAME) -> None:
        try:
            if MongoDBClient is None:
                mongo_db_url = os.getenv(client_url)
                if mongo_db_url is None:
                    raise Exception("Environment key : mongo_db_url is not set.")
                
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url , tlsCAFile = ca)
                     
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        
        except Exception as e:
            raise CustomException(e ,sys)
                