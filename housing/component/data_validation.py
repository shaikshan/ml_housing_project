

from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import os,sys
import pandas  as pd
from housing.util.util import *
from housing.constant import *
from scipy.stats import ks_2samp
import json

class DataValidation:
    

    def __init__(self, data_validation_config:DataValidationConfig,
        data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*30}Data Valdaition log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml_file(file_path=data_validation_config.schema_file_path)
        except Exception as e:
            raise HousingException(e,sys) from e


    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
        except Exception as e:
            raise HousingException(e,sys) from e


    def is_train_test_file_exists(self)->bool:
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available =  is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")
            
            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file}" \
                    "is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_previous_data(self):
        try:
            pass
        except Exception as e:
            raise HousingException(e,sys) from e

    
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False
            
            #Assigment validate training and testing dataset using schema file
            #1. Number of Column
            #2. Check the value of ocean proximity 
            # acceptable values     <1H OCEAN
            # INLAND
            # ISLAND
            # NEAR BAY
            # NEAR OCEAN
            #3. Check column names


            validation_status = True
            return validation_status 
        except Exception as e:
            raise HousingException(e,sys) from e

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self.schema_config['columns'])

            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise HousingException(e,sys) from e
        
    def is_numeric_columns_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = self.schema_config['numerical_columns']
            dataframe_columns = dataframe.columns


            missing_numerical_columns = []

            numerical_columns_present = True
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(num_column)

            logging.info(f"Missing numerical columns:[{missing_numerical_columns}]")
            return numerical_columns_present

        except Exception as e:
            raise HousingException(e,sys) from e
        
    def get_data_drift_report(self,train_df,test_df,threshold=0.5)->bool:
        try:
            report ={}
            status = True
            True_count = 0
            False_count = 0
            for column in train_df.columns:
                d1 = train_df[column]
                d2 = test_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                if is_found is True:
                    True_count +=1
                else:
                    False_count +=1
                if True_count >= len(train_df.columns)//2:
                    status =False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})

            logging.info(f"True count of data drift:[{True_count}]")
            logging.info(f"False count of data drift:[{False_count}]")
            logging.info(f"Difference of True and count and Number of columns:{True_count-len(train_df.columns)}")

            #Creating directory
            drift_report_file_path = self.data_validation_config.report_file_path
            dir_name = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_name,exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path,data=report)

            return status       
        except Exception as e:
            raise HousingException(e,sys) from e
        
    
    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            error_message = ""
            #Reading train and test data into DataFrame
            train_df,test_df = self.get_train_and_test_df()

            #Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                error_message = f"{error_message} Train dataframe does not contain all columns \n"
            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain all columns \n"

            #Validate numerical columns
            status = self.is_numeric_columns_exist(dataframe=train_df)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain numerical columns"
            status = self.is_numeric_columns_exist(dataframe=test_df)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain numerical columns"

            if len(error_message)>0:
                raise Exception(error_message)
            
            #Lets Check data drift
            status = self.get_data_drift_report(train_df=train_df,test_df=test_df)

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")
        



