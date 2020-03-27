import pandas as pd
import numpy as np
import warnings
import os



def get_result(id):
    pred = False
    predictions = pd.read_csv('data/predictions.csv')
    if not predictions[predictions['material_id'] == id].empty:
        result = predictions[predictions['material_id'] == id].values[0].tolist()
        pred = True
    else:
        elastic = pd.read_csv('data/elastic.csv')
        if not elastic[elastic['material_id'] == id].empty:
            result = elastic[elastic['material_id'] == id].values[0].tolist()
        else:
            raise ValueError('invalid id')
    return result, pred
