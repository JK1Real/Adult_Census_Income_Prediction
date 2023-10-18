import sys
import os
from dataclasses import dataclass



import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler


sys.path.append('C:\\Users\\HP\\Desktop\\projects\\Adult_census_Income_Prediction\\src')

from src.utils import save_obj
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pk1")

class DataTransformation :
     def __init__(self) :
          self.data_tranformation_config = DataTransformationConfig()

     def get_data_transformer_object(self) :
          """
          This function is responsible for data transformation
          """
          try :
               numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']
               categorical_columns = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'country']

               num_pipeline = Pipeline(
                    steps = [
                         ("imputer",SimpleImputer(strategy="median")),
                         ("scaler",StandardScaler())
                    ]
               )
               cat_pipeline = Pipeline(
                    steps= [
                         ("imputer",SimpleImputer(strategy="most_frequent")),
                         ("scaler",StandardScaler())
                    ]
               )

               logging.info("Numerical columns standard scaling completed")
               logging.info("Categorical columns encoding completed")

               preprocessor = ColumnTransformer(
                    [
                         ("num_pipeline",num_pipeline,numerical_columns) ,
                         ("cat_pipeline",cat_pipeline,categorical_columns)
                    ]
               )

               return preprocessor
          
          except Exception as e :
               raise CustomException(e,sys)
        

     def initiate_data_transformation(self,train_path,test_path) :
          
          try :
               train_df = pd.read_csv(train_path)
               test_df = pd.read_csv(test_path)

               train_df=train_df.replace(' ?',np.nan)
               test_df=test_df.replace(' ?',np.nan)

             
               train_df['workclass']=train_df['workclass'].fillna(train_df['workclass'].mode()[0])
               train_df['occupation']=train_df['occupation'].fillna(train_df['occupation'].mode()[0])
               train_df['country']=train_df['country'].fillna(train_df['country'].mode()[0])

               test_df['workclass']=test_df['workclass'].fillna(test_df['workclass'].mode()[0])
               test_df['occupation']=test_df['occupation'].fillna(test_df['occupation'].mode()[0])
               test_df['country']=test_df['country'].fillna(test_df['country'].mode()[0])

               # Mapping 'salary' values to numeric values
               train_df['salary'] = train_df['salary'].map({' <=50K': 0, ' >50K': 1})
               test_df['salary'] = test_df['salary'].map({' <=50K': 0, ' >50K': 1})

               logging.info(train_df.isnull().sum())

               logging.info(f" Read train and test data completed {train_df.shape,test_df.shape} ")
               logging.info(" Obtaining preprocessing object ")


               
               preprocessing_obj = self.get_data_transformer_object()

               target_column = "salary"
               numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']
               categorical_columns = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'country']

               logging.info(train_df.head())
               logging.info(test_df.head())            




               

               logging.info(test_df["salary"].value_counts())
                # Apply LabelEncoder to categorical columns
               label_encoder = LabelEncoder()
               for col in categorical_columns:
                    train_df[col] = label_encoder.fit_transform(train_df[col])
                    test_df[col] = label_encoder.transform(test_df[col])

               
               input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
               target_feature_train_df=train_df[target_column]
               
               
               input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
               target_feature_test_df=test_df[target_column]


               logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe." )
               
               input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
               input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
               

               train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]
               test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
               
             
               save_obj(
                    file_path = self.data_tranformation_config.preprocessor_obj_file_path ,
                    obj = preprocessing_obj
               )


               return (
                    train_arr,
                    test_arr,
                    self.data_tranformation_config.preprocessor_obj_file_path
               )
          
          except Exception as e:
              raise CustomException(sys,e)
