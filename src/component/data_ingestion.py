#Lire les données de n'importe quelle source de données 
#impoter OS, sys,  custonException  et logging car on utilisera l'exception

import os 
import sys
from src.logger import logging
from src.exception import CustomException


#Importer les differentes bibliothèque 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig
from src.component.model_trainer import ModelTrainerConfig
from src.component.model_trainer import ModelTrainer

#1 LIRE LES DONNEES
#Input: creer une classe qui permet de stocker les chemins
@dataclass #pour eviter de creer def init 
#Specifier les chemins 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")#chemin pour sauvegarder les données d'entrainement après split
    test_data_path: str=os.path.join('artifacts', "test.csv")#chemin pour sauvegarder les données de test après split
    raw_data_path: str=os.path.join('artifacts', "raw.csv")#save les données brutes avant split

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    # la fonction suivante permet de lire le fichier .csv, sauvegarder en brute, découpage et sauvegarder les résultats dans des fichiers .csv
    def initiate_data_ingestion(self): 
        #Simple method 
        logging.info("Entered the data ingestion method or component")
        try: 
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read dataset as dataframe')


            #creer le dossier pour stocker les fichiers
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #sauvegarder les données brutes
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split Train/test
            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            # save train_set dans le dossier artifact
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # save test_set dans le dossier artifact
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
    
# executer le code uniquement si ce fichier sera lancé, pas de import dans d'autre fichier 
if __name__ =="__main__":
    #Combine Data Ingestion
    obj=DataIngestion()
    train_data, test_data, raw_data= obj.initiate_data_ingestion()

    #Combine Data Transformation
    data_transformation= DataTransformation()
    train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

