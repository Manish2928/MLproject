import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join("artifect","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train ,y_train,x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],

                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()

            }

            params = {
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                },

                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Gradient Boosting":{
                    
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],

                },

                "Linear regression":{},

                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11]

                },

                "XGBRegressor":{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001],

                },

                # "CatBoosting Regressor":{
                #     'depth':[6,8,10],
                #     'iterations': [30,50,100],
                #     'learning_rate':[0.01,0.05,0.1],

                # },

                "AdaBoost Regressor":{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate':[0.01,0.05,0.1],

                },
            }

            
            model_reprot:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test, models=models,params=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_reprot.values()))

            ## To get best model name from dict
            best_model_name = list(model_reprot.keys())[list(model_reprot.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise customException("No Best model Found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square



        except Exception as e:
            raise customException(e,sys)
        
