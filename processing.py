from pymatgen import MPRester, Composition
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import argparse
from gemmi import cif
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates
from matminer.featurizers.structure import DensityFeatures
import joblib
import os
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
import predict
from predict import parser

all_properties = ["material_id", "elements", "pretty_formula", "structure", "cif"]

# Make sure that you have the Materials API key. Put the key in the call to
# MPRester if needed, e.g, MPRester("MY_API_KEY")
mpr = MPRester("nbAXKL1lCrVP88v2I")

def model_prediction(id):
    for f in os.listdir('cifs'):
        if f.endswith('.cif') or f.endswith('.csv'):
            os.remove(os.path.join('cifs', f))
    data = mpr.query(criteria={"material_id": id},properties=all_properties)
    X = pd.DataFrame(data)
    for i, row in X.iterrows():
        item = cif.read_string(row["cif"])
        item.write_file('cifs/' + str(row["material_id"]) + '.cif')
    df = pd.DataFrame({'material_id':[id], 'value':[0]})
    df.to_csv('cifs/id_prop.csv', index=False, header=False)

    X = StrToComposition().featurize_dataframe(X, "pretty_formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    X = ep_feat.featurize_dataframe(X, col_id="composition")
    X = CompositionToOxidComposition().featurize_dataframe(X, "composition")
    os_feat = OxidationStates()
    X = os_feat.featurize_dataframe(X, "composition_oxid")
    den_feat = DensityFeatures()
    X = den_feat.featurize_dataframe(X, "structure", ignore_errors=True)  # input the structure column to the featurizer
    result = []
    result.append(id)
    result.append(X['pretty_formula'].values[0])
    X.drop(["material_id", "pretty_formula", "cif", "structure", "elements", "composition", "composition_oxid"], axis=1, inplace=True)

    #cgcnn prediction

    best_mae_error = 1e10




    x_K = joblib.load("models/xgb_K.joblib.dat")
    x_G = joblib.load("models/xgb_G.joblib.dat")
    x_p = joblib.load("models/xgb_p.joblib.dat")
    model_K = tf.keras.models.load_model('models/stacked_K.h5')
    model_G = tf.keras.models.load_model('models/stacked_G.h5')
    model_p = tf.keras.models.load_model('models/stacked_p.h5')


    args = parser.parse_args(['models/cgcnn_K.pth.tar', 'cifs'])
    if os.path.isfile(args.modelpath):
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    best_mae_error = 1e10
    predict.main(args, model_args, best_mae_error)

    temp = pd.read_csv('test_results.csv', header=None)
    temp = temp.rename(columns={0: 'material_id', 1:'none', 2:'cgcnn'})

    cgcnn = temp['cgcnn'].values
    pred = x_K.predict(X)
    y_pred = model_K.predict(np.stack((cgcnn,pred), axis=-1))
    result.append(y_pred[0][0])

    args = parser.parse_args(['models/cgcnn_G.pth.tar', 'cifs'])
    if os.path.isfile(args.modelpath):
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    predict.main(args, model_args, best_mae_error)

    temp = pd.read_csv('test_results.csv', header=None)
    temp = temp.rename(columns={0: 'material_id', 1:'none', 2:'cgcnn'})

    cgcnn = temp['cgcnn'].values
    pred = x_G.predict(X)
    y_pred = model_G.predict(np.stack((cgcnn,pred), axis=-1))
    result.append(y_pred[0][0])

    args = parser.parse_args(['models/cgcnn_p.pth.tar', 'cifs'])
    if os.path.isfile(args.modelpath):
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    predict.main(args, model_args, best_mae_error)

    temp = pd.read_csv('test_results.csv', header=None)
    temp = temp.rename(columns={0: 'material_id', 1:'none', 2:'cgcnn'})

    cgcnn = temp['cgcnn'].values
    pred = x_p.predict(X)
    y_pred = model_p.predict(np.stack((cgcnn,pred), axis=-1))
    result.append(y_pred[0][0])

    os.remove('test_results.csv')

    return result
