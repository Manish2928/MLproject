import os
import sys
from src.exception import customException
from src.logger import logging
import dill

import numpy as np
import pandas as pd


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)