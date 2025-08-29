import sys
from dataclasses import dataclass # pour eviter Init
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer #permet de faire des transformation pour les colonnes
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline #enchainer les étapes
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os 
from src.utils import save_object

#But : stocker les chemins/configs de cette étape.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "processor.pkl")

class DataTransformation:
    #But : instancier la config pour pouvoir accéder au chemin de sortie 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    
    def get_data_transformation_object(self):
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler())
                ] 
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    (("scaler", StandardScaler(with_mean=False)) )

                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info("numercial columns standard scaling completed")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info("categorical columns encoding completed")

            #Combiner les deux pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # 1- Load
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # 2 -  Preprocessing
            preprocessing_obj = self.get_data_transformation_object() #renvoie le preprocesseur

            # 3 - Split X/y
            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            #4 - Fit/transform 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 5 - Concat X_transformed + y -Concatiner horizontalement X_train transformé et y_train en un seul tableau numpy-
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)

            ]

            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                
            ]

            logging.info(f"Saved preprocessing object.")

            # 6 -save preprocessor
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            # 7 - Return packed outputs
            return[
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            ]
        except Exception as e: 
           raise CustomException(e,sys)