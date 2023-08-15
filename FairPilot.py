from __future__ import annotations

#from sklearnex import patch_sklearn 

#patch_sklearn()

import base64
import os
import json
import pickle
import uuid
import re
import time
import streamlit as st
from math import floor
import pandas as pd
import numpy as np
#import tensorflow as tf
from tensorflow import keras
import plotly.io as pio
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from statistics import mean, stdev, variance
from scipy.stats import sem
from SingleModelPareto import SingleModelParetoDT
from SingleModelPareto2 import SingleModelParetoSVC
from SingleModelPareto3 import SingleModelParetoLR
from SingleModelPareto4 import SingleModelParetoRF
from SingleModelPareto5 import SingleModelParetoNN
import FairnessCriterions as fc
import Pareto as pt
import MultiModelPF as mmpf
import MultiModelPF_MOBO as mmpf_bo
import superimposedPF as SPF
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ConfigSpace import ForbiddenEqualsClause, ForbiddenAndConjunction

import warnings
import matplotlib.pyplot as plt
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.facade.abstract_facade import AbstractFacade
from smac.multi_objective.parego import ParEGO
from smac.initial_design.latin_hypercube_design import LatinHypercubeInitialDesign

from IPython.display import HTML

def fairness(X1, y1, X2, y2, model, isNN) :
    
    if isNN == True :
        pred1_NN = model.predict(X1, verbose=0) 
        pred1 = np.argmax(pred1_NN, axis=1)
        pred2_NN = model.predict(X2, verbose=0) 
        pred2 = np.argmax(pred2_NN, axis=1)
    else :
        pred1 = model.predict(X1)
        pred2 = model.predict(X2)

    m1 = confusion_matrix(y1, pred1)
    TN1, FP1, FN1, TP1 = m1.ravel()
    tot1 = TN1 + FP1 + FN1 + TP1
    acc1 = (TN1+TP1)/tot1
    ppv1 = TP1/(TP1+FP1)
    FPR1 = FP1/(TP1+FP1)
    NPV1 = TN1/(TN1+FN1)
    FOR1 = FN1/(TN1+FN1)
    TPR1 = TP1/(TP1+FN1)
    FNR1 = FN1/(TP1+FN1)
    TNR1 = TN1/(TN1+FP1)
    FPR1 = FP1/(TN1+FP1)
    stp1 = (TP1+FP1)/tot1
    GRAD1 = (FP1 + TP1)/tot1

    m2 = confusion_matrix(y2, pred2)
    TN2, FP2, FN2, TP2 = m2.ravel()
    tot2 = TN2 + FP2 + FN2 + TP2
    acc2 = (TN2+TP2)/tot2
    ppv2 = TP2/(TP2+FP2)
    FPR2 = FP2/(TP2+FP2)
    NPV2 = TN2/(TN2+FN2)
    FOR2 = FN2/(TN2+FN2)
    TPR2 = TP2/(TP2+FN2)
    FNR2 = FN2/(TP2+FN2)
    TNR2 = TN2/(TN2+FP2)
    FPR2 = FP2/(TN2+FP2)
    stp2 = (TP2+FP2)/tot2
    GRAD2 = (FP2 + TP2)/tot2
    
    prp = abs(ppv1 - ppv2)
    prE = abs(FPR1 - FPR2)
    Eop = abs(FNR1 - FNR2)
    Acc = abs(acc1 - acc2)
    Eq1 = 0.5*(abs(TPR1 - TPR2) + abs(FPR1 - FPR2))
    Pty = abs(GRAD1 - GRAD2)
    
    return prp, prE, Eop, Acc, Eq1, Pty

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(8, 232, 150);
                text-align:center;
                color:#000000;
                padding-top: 8px;
                padding-bottom: 8px;
                padding-left: 20px;
                padding-right: 20px;
                position: relative;
                text-decoration: none;
                border-radius: 6px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(200, 200, 200);
                border-image: initial;
                left: 39%;

            }} 
            #{button_id}:hover {{
                border-color: rgb(8, 130, 50);
                color: rgb(8, 130, 50);
                background-color: rgb(8, 232, 135);

            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(8, 232, 135);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

#################################
#################################
###--------- SIDEBAR ---------###
#################################
#################################

st.sidebar.header('FairPilot Settings')

st.sidebar.subheader('Optimization Algorithm')
opt_method = st.sidebar.selectbox('Choose the appropriate Optimization Algorithm',
                                  ('Grid Search','MOBO'),
                                    help="Grid search: Build a custom grid and try out every possible combination. Allows for a better interpretation of Hyperparameter's influence on Fairness. MOBO: The optimization is performed fully automatically in a smarter way. Outputs results closer to the optimal values.")

st.sidebar.subheader('Choose Learning Methods')

checkDT = st.sidebar.checkbox('Decision Tree Classifier')
if checkDT and opt_method=='Grid Search':
    with st.sidebar.expander('**Set HP space:**'):
        criterion = st.multiselect('Criterion', ['gini', 'entropy', 'log_loss'], ['gini'])
        class_weight = list(st.multiselect('Class-Weight', [None, 'balanced'], [None]))
        max_depth_step = st.number_input('Max depth - Step size',1,5)
        max_depth = np.array(st.slider('Max depth', 5, 100, [5, 25], max_depth_step))
        max_features = np.array(st.multiselect('Max-features', [None, 'sqrt', 1], [None]))
        min_samples_split_step = st.number_input('Min-samples split - Step size',1,5)
        min_samples_split = np.array(st.slider('Min-samples split', 2, 20, [2, 10], min_samples_split_step))
        min_samples_leaf_step = st.number_input('Min-samples leaf - Step size',1,5)
        min_samples_leaf = np.array(st.slider('Min-samples leaf', 1, 20, [5, 10], min_samples_leaf_step))

        min_samples_leaf = np.arange(min_samples_leaf[0], min_samples_leaf[1]+min_samples_leaf_step, min_samples_leaf_step)
        if min_samples_leaf[-1] < 20:
            min_samples_leaf = np.append(min_samples_leaf, 20)
        elif min_samples_leaf[-1] > 20:
            min_samples_leaf[-1] = 20
        min_samples_split = np.arange(min_samples_split[0], min_samples_split[1]+min_samples_split_step, min_samples_split_step)
        if min_samples_split[-1] < 20:
            min_samples_split = np.append(min_samples_split, 20)
        elif min_samples_split[-1] > 20:
            min_samples_split[-1] = 20
        max_depth = np.arange(max_depth[0], max_depth[1]+max_depth_step, max_depth_step)
        if max_depth[-1] > 100:
            max_depth[-1] = 100

checkSVC = st.sidebar.checkbox('Support Vector Classifier')
if checkSVC and opt_method=='Grid Search':
    with st.sidebar.expander('**Set HP space:**'):
        penalties = st.multiselect('Penalty', ['l1', 'l2'])
        Cs = np.array(list(map(float, st.multiselect('C ', ['1000', '100', '10', '1', 
                            '0.1', '0.01', '0.001'], ['1']))))
        losses = st.multiselect('Loss', ['hinge', 'squared_hinge'])
        fit_intercepts = st.multiselect('Fit intercept', [True, False])
        intercept_scalings = np.array(st.slider('Intercept scaling', 1, 10, [1, 10]))

checkLR = st.sidebar.checkbox('Logistic Regression')
if checkLR and opt_method=='Grid Search':
    with st.sidebar.expander('**Set HP space:**'):
        LR_penalties = st.multiselect('Penalty', ['none', 'l2'], ['none'])
        LR_Cs = np.array(list(map(float, st.multiselect('C', ['1000', '100', '10', '1', 
                            '0.1', '0.01', '0.001'], ['1']))))

checkRF = st.sidebar.checkbox('Random Forest Classifier')
if checkRF and opt_method=='Grid Search':
    with st.sidebar.expander('**Set HP space:**'):
        bootstrap = st.multiselect('Bootstrap', [True, False], [True])
        max_samples = np.array(list(map(float, st.multiselect('Max bootstrap samples', ['0.5', '0.6', '0.7', '0.8', 
                            '0.0', '1.0'], ['1.0']))))
        RF_criterion = st.multiselect('Criterion ', ['gini', 'entropy'], ['gini'])
        RF_class_weight= st.multiselect('Class-Weight ', [None, 'balanced'], [None])
        RF_max_depth_step = st.number_input('Max depth - Step size ',1,5)
        RF_max_depth = np.array(st.slider('Max depth ', 2, 100, [5, 25], RF_max_depth_step))
        RF_max_features = st.multiselect('Max-features ', [None, 'sqrt', 1], ['sqrt'])
        RF_min_samples_split_step = st.number_input('Min-samples split - Step size ',1,5)
        RF_min_samples_split = np.array(st.slider('Min-samples split ', 2, 20, [2, 10], RF_min_samples_split_step))
        RF_min_samples_leaf_step = st.number_input('Min-samples leaf - Step size ',1,5)
        RF_min_samples_leaf = np.array(st.slider('Min-samples leaf ', 1, 20, [5, 10], RF_min_samples_leaf_step))

        RF_min_samples_leaf = np.arange(RF_min_samples_leaf[0], RF_min_samples_leaf[1]+RF_min_samples_leaf_step, RF_min_samples_leaf_step)
        if RF_min_samples_leaf[-1] < 20:
            RF_min_samples_leaf = np.append(RF_min_samples_leaf, 20)
        elif RF_min_samples_leaf[-1] > 20:
            RF_min_samples_leaf[-1] = 20
        RF_min_samples_split = np.arange(RF_min_samples_split[0], RF_min_samples_split[1]+RF_min_samples_split_step, RF_min_samples_split_step)
        if RF_min_samples_split[-1] < 20:
            RF_min_samples_split = np.append(RF_min_samples_split, 20)
        elif RF_min_samples_split[-1] > 20:
            RF_min_samples_split[-1] = 20
        RF_max_depth = np.arange(RF_max_depth[0], RF_max_depth[1]+RF_max_depth_step, RF_max_depth_step)
        if RF_max_depth[-1] < 100:
            RF_max_depth = np.append(max_depth, 100)
        elif max_depth[-1] > 100:
            RF_max_depth[-1] = 100

if opt_method == 'Grid Search':
    checkNN = st.sidebar.checkbox('Neural Network')
    if checkNN and opt_method=='Grid Search':
        with st.sidebar.expander('**Set HP space:**'):
            L1_nodes = np.array(list((map(int, st.multiselect('Layer 1 nodes', ['64', '128', '256'], ['64'])))))
            L2_nodes = np.array(list((map(int, st.multiselect('Layer 2 nodes', ['64', '128', '256'], ['64'])))))
            L1_dropout_rates = np.array(st.slider('Layer 1 dropout', 0.1, 0.5, [0.2, 0.4], 0.1))
            L2_dropout_rates = np.array(st.slider('Layer 2 dropout', 0.1, 0.5, [0.2, 0.4], 0.1))
            batch_sizes = np.array(list((map(int, st.multiselect('Batch size', ['4', '8', '16', '32'], ['8'])))))
            epochs = np.array(list((map(int, st.multiselect('Epochs', ['5', '10', '15'], ['10'])))))

st.sidebar.write('---')

st.sidebar.subheader('Choose Fairness Metrics')
prp_ok = st.sidebar.checkbox('Predictive Parity',
                    help='Probability of a positive outcome should be the same for both classes.')
prE_ok = st.sidebar.checkbox('Predictive Equality', 
                    help='The fraction of false positive (or true negative) predictions should be the same for both classes.')
Eop_ok = st.sidebar.checkbox('Equal Opportunity',
                    help='The fraction of false negative (or true positive) predictions should be the same for both classes.')
Acc_ok = st.sidebar.checkbox('Accuracy Equality',
                    help='Prediction accuracy should be the same for both classes.')
Eq1_ok = st.sidebar.checkbox('Equalized Odds',
                    help='The fraction of false positive AND true positive predictions should be the same for both classes.')
Pty_ok = st.sidebar.checkbox('Statistical Parity',
                    help='The ratio of positive predictions should be the same regardless of the class.')

notions = ['', '', '', '', '', '']
notions2 = []

if prp_ok: 
    notions[0] = 'predictive parity'
    notions2.append('predictive parity')
if prE_ok: 
    notions[1] = 'predictive equality'
    notions2.append('predictive equality')
if Eop_ok: 
    notions[2] = 'equal opportunity'
    notions2.append('equal opportunity')
if Acc_ok: 
    notions[3] = 'accuracy equality'
    notions2.append('accuracy equality')
if Eq1_ok: 
    notions[4] = 'equalized odds'
    notions2.append('equalized odds')
if Pty_ok: 
    notions[5] = 'statistical parity'
    notions2.append('statistical parity')

st.sidebar.write('---')

st.sidebar.subheader('Advanced Settings')
split_size = st.sidebar.slider('**Data Split Ratio (% of Training Set)**', 50, 90, 80, 5)
n_seeds = st.sidebar.slider('**Number of Repetitions**', 2, 20, 5, 1)
scl_type = st.sidebar.selectbox('**Data Scaling**',('None', 'Standardization', 'Normalization'))
imp_type = st.sidebar.selectbox('**Data Imputation**',('None', 'Median', 'Mean'))

###################################
###################################
###--------- MAIN MENU ---------###
###################################
###################################


st.write("# FairPilot")

st.subheader('Upload Your Dataset')
file = st.file_uploader('For optimal results, consider uploading a pre-processed dataset.', 
                 type= ['xlsx', 'csv'], key=None, help='While FairPilot is able to perform some steps of data preprocessing, performing this step separately allows more control over the results.')

if file is not None:
    df = pd.read_excel(file) # Load the dataset
    st.subheader('Dataset')
    st.markdown("Here's a sneak peek of your dataset:")

    st.write(df.head(5)) # First five rows of the dataset

    y_name = st.selectbox('Choose output variable', df.columns)
    X_sen = st.selectbox('Choose sensitive attribute', df.columns)

    df_MODEL = []
    if prp_ok : df_MODEL_prp = []
    if prE_ok : df_MODEL_prE = []
    if Eop_ok : df_MODEL_Eop = []
    if Acc_ok : df_MODEL_Acc = []
    if Eq1_ok : df_MODEL_Eq1 = []
    if Pty_ok : df_MODEL_Pty = []

    for i in range(5) :
        edf = pd.DataFrame()
        df_MODEL.append(edf)
        if prp_ok : 
            df_MODEL_prp.append(edf)
        if prE_ok : 
            df_MODEL_prE.append(edf)
        if Eop_ok : 
            df_MODEL_Eop.append(edf)
        if Acc_ok : 
            df_MODEL_Acc.append(edf)
        if Eq1_ok : 
            df_MODEL_Eq1.append(edf)
        if Pty_ok : 
            df_MODEL_Pty.append(edf)

    m = st.markdown("""
    <style >
    .stDownloadButton, div.stButton {text-align:center}
    .stDownloadButton button, div.stButton > button:first-child {
        background-color: rgb(8, 232, 150);
        color:#000000;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 5px;
        border-width: 1px;
        border-style: solid;
        border-color: rgb(200, 200, 200);
    }
    .stDownloadButton button:hover, div.stButton > button:hover {
        color:#000000;
        border-color: rgb(8, 130, 50);
        color: rgb(8, 130, 50);
        background-color: rgb(8, 232, 135);
    } 
    .stDownloadButton button:focus, div.stButton > button:focus {
        background-color: rgb(8, 150, 135);
        color:#FFFFFF;
    }
    </style>""", unsafe_allow_html=True)

    if st.button('**START**'):
        if y_name and X_sen:
            dataset = df.dropna(subset = [y_name])
            if imp_type=='None':
                X = dataset.dropna().drop(columns = [y_name])
                y = dataset.dropna().loc[:, y_name]
            elif imp_type=='Median':
                X = dataset.fillna(dataset.median()).drop(columns = [y_name])
                y = dataset.loc[:, y_name]
            else:
                X = dataset.fillna(dataset.mean()).drop(columns = [y_name])
                y = dataset.loc[:, y_name]
            X_numerical_features = X[[col for col in X.columns if len(np.unique(X[col])) > 2]]
            X_categorical_features = X[[col for col in X.columns if len(np.unique(X[col])) == 2]]   
            if scl_type=='Normalization':
                ss = StandardScaler()
                X_numerical_features_scaled = ss.fit_transform(X_numerical_features)
                X_numerical_features_scaled = pd.DataFrame(X_numerical_features_scaled, columns = X_numerical_features.columns)
                X_scaled = pd.concat([X_numerical_features_scaled, X_categorical_features], axis=1)
            elif scl_type=='Standardization':
                ss = MinMaxScaler()
                X_numerical_features_scaled = ss.fit_transform(X_numerical_features)
                X_numerical_features_scaled = pd.DataFrame(X_numerical_features_scaled, columns = X_numerical_features.columns)
                X_scaled = pd.concat([X_numerical_features_scaled, X_categorical_features], axis=1)
            else:
                X_scaled = X


            #######################################
            #######################################
            ###--------- DECISION TREE ---------###
            #######################################
            #######################################


            if checkDT and opt_method=='Grid Search':

                test_scores = []
                test_scores_privileged = []
                test_scores_not_privileged = []

                mean_test_scores = []
                mean_test_scores_privileged = []
                mean_test_scores_not_privileged = []

                sem_test_scores = []

                fairness_metrics = []

                mean_prp = []
                mean_prE = []
                mean_Eop = []
                mean_Acc = []
                mean_Eq1 = []
                mean_Pty = []

                sem_prp = []
                sem_prE = []
                sem_Eop = []
                sem_Acc = []
                sem_Eq1 = []
                sem_Pty = []

                hyp_min_samples_leaf = []
                hyp_min_samples_split = []
                hyp_max_features = []
                hyp_criterions = []
                hyp_class_weights = []
                hyp_max_depth = []

                output = []
                i = 0
                total_combinations = len(class_weight) * len(criterion) * len(max_features) * min_samples_split.size * min_samples_leaf.size * max_depth.size

                progress_text = "Fitting decision trees..."
                my_bar = st.progress(0, text=progress_text)

                for w in class_weight :
                    for depth in max_depth :
                        for c in criterion :
                            for features in max_features :
                                for samples_split in min_samples_split :
                                    for samples_leaf in min_samples_leaf :
                                        for seed in range(n_seeds) :
                                            
                                            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=seed, train_size=split_size)
                                            X_privileged_test = X_test[(X_test[X_sen] > 0)]
                                            X_not_privileged_test = X_test[(X_test[X_sen] == 0)]
                                            y_privileged_test = y_test[(X_test[X_sen] > 0)]
                                            y_not_privileged_test = y_test[(X_test[X_sen] == 0)]
                                            dtr = DecisionTreeClassifier(class_weight=w, max_depth=depth, criterion=c, 
                                                                         max_features=features, min_samples_split=samples_split, 
                                                                         min_samples_leaf=samples_leaf, random_state=0)
                                            dtr.fit(X_train, y_train)
                                            test_scores.append(dtr.score(X_test, y_test))
                                            test_scores_privileged.append(dtr.score(X_privileged_test, y_privileged_test))
                                            test_scores_not_privileged.append(dtr.score(X_not_privileged_test, y_not_privileged_test))

                                            output.append(fc.fairness(X_privileged_test, y_privileged_test, X_not_privileged_test, y_not_privileged_test, dtr, False))

                                        fairness_metrics = np.array(output)

                                        ## Fairness metrics
                                        mean_prp.append(fairness_metrics[:, 0].mean())
                                        mean_prE.append(fairness_metrics[:, 1].mean())
                                        mean_Eop.append(fairness_metrics[:, 2].mean())
                                        mean_Acc.append(fairness_metrics[:, 3].mean())
                                        mean_Eq1.append(fairness_metrics[:, 4].mean())
                                        mean_Pty.append(fairness_metrics[:, 5].mean())

                                        sem_prp.append(sem(fairness_metrics[:, 0]))
                                        sem_prE.append(sem(fairness_metrics[:, 1]))
                                        sem_Eop.append(sem(fairness_metrics[:, 2]))
                                        sem_Acc.append(sem(fairness_metrics[:, 3]))
                                        sem_Eq1.append(sem(fairness_metrics[:, 4]))
                                        sem_Pty.append(sem(fairness_metrics[:, 5]))

                                        ## Accuracy scores
                                        mean_test_scores.append(mean(test_scores))
                                        mean_test_scores_privileged.append(mean(test_scores_privileged))
                                        mean_test_scores_not_privileged.append(mean(test_scores_not_privileged))

                                        sem_test_scores.append(sem(test_scores))

                                        ## Hyperparameters
                                        hyp_min_samples_leaf.append(samples_leaf)
                                        hyp_min_samples_split.append(samples_split)
                                        hyp_max_features.append(features)
                                        hyp_criterions.append(c)
                                        hyp_class_weights.append(w)
                                        hyp_max_depth.append(depth)

                                        ## Reset for each hyperparameter combination
                                        output = []
                                        test_scores = []
                                        test_scores_privileged = []
                                        test_scores_not_privileged = []
                                        i = i + 1
                                        my_bar.progress(i/total_combinations, text=progress_text)
                                
                st.subheader('Decision Tree Classifier: Results')

                solutions_space = {
                    'min_samples_leaf': hyp_min_samples_leaf, 
                    'min_samples_split': hyp_min_samples_split,
                    'max_features': hyp_max_features,
                    'max_depth': hyp_max_depth,
                    'criterion': hyp_criterions,
                    'class_weight': hyp_class_weights,
                    'Accuracy': mean_test_scores,
                    'Accuracy std': sem_test_scores,
                    'Predictive parity': mean_prp,
                    'Predictive parity SEM': sem_prp,
                    'Predictive equality': mean_prE,
                    'Predictive equality SEM': sem_prE,
                    'Equal opportunity': mean_Eop,
                    'Equal opportunity SEM': sem_Eop,
                    'Accuracy equality': mean_Acc,
                    'Accuracy equality SEM': sem_Acc,
                    'Equalized odds': mean_Eq1,
                    'Equalized odds SEM': sem_Eq1,
                    'Statistical parity': mean_Pty,
                    'Statistical parity SEM': sem_Pty,
                }

                df_dt = pd.DataFrame(data=solutions_space)
                df_dt.replace(to_replace=[None], value='None', inplace=True)
                df_MODEL[0] = df_dt
                st.write(df_dt.head(10))

                csv = convert_df(df_dt)

                download_button_str = download_button(csv, "dt_full_table.csv", 'Download Table', pickle_it=False)
                st.markdown(download_button_str, unsafe_allow_html=True)

                st.write('---')

                if prp_ok:
                    df_dt_prp = pt.pareto_frontier(Xs=df_dt['Accuracy'], Ys=df_dt['Predictive parity'], name='Predictive parity', maxX=True, maxY=True)
                if prE_ok:
                    df_dt_prE = pt.pareto_frontier(Xs=df_dt['Accuracy'], Ys=df_dt['Predictive equality'], name='Predictive equality', maxX=True, maxY=True)
                if Eop_ok:
                    df_dt_Eop = pt.pareto_frontier(Xs=df_dt['Accuracy'], Ys=df_dt['Equal opportunity'], name='Equal opportunity', maxX=True, maxY=True)
                if Acc_ok:    
                    df_dt_Acc = pt.pareto_frontier(Xs=df_dt['Accuracy'], Ys=df_dt['Accuracy equality'], name='Accuracy equality', maxX=True, maxY=True)
                if Eq1_ok:
                    df_dt_Eq1 = pt.pareto_frontier(Xs=df_dt['Accuracy'], Ys=df_dt['Equalized odds'], name='Equalized odds', maxX=True, maxY=True)
                if Pty_ok:
                    df_dt_Pty = pt.pareto_frontier(Xs=df_dt['Accuracy'], Ys=df_dt['Statistical parity'], name='Statistical parity', maxX=True, maxY=True)

                mean_test_scores_np = np.array(mean_test_scores)
                sem_test_scores_np = np.array(sem_test_scores)
                prp_np = np.array(mean_prp)
                prE_np = np.array(mean_prE)
                Eop_np = np.array(mean_Eop)
                Acc_np = np.array(mean_Acc)
                Eq1_np = np.array(mean_Eq1)
                Pty_np = np.array(mean_Pty)

                sem_prp_np = np.array(sem_prp)
                sem_prE_np = np.array(sem_prE)
                sem_Eop_np = np.array(sem_Eop)
                sem_Acc_np = np.array(sem_Acc)
                sem_Eq1_np = np.array(sem_Eq1)
                sem_Pty_np = np.array(sem_Pty)

                if prp_ok:
                    st.subheader('DT - Predictive Parity')
                    df_dt_prp = SingleModelParetoDT(df_dt_prp, prp_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_max_depth, df_dt, 'Predictive parity')
                    df_dt_prp.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prp[0] = df_dt_prp
                    st.write(df_dt_prp)
                    csv = convert_df(df_dt_prp)

                    download_button_str = download_button(csv, "dt_pareto_prp.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if prE_ok:
                    st.subheader('DT - Predictive Equality')
                    df_dt_prE = SingleModelParetoDT(df_dt_prE, prE_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_max_depth, df_dt, 'Predictive equality')
                    df_dt_prE.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prE[0] = df_dt_prE
                    st.write(df_dt_prE)
                    csv = convert_df(df_dt_prE)
                    download_button_str = download_button(csv, "dt_pareto_prE.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eop_ok:
                    st.subheader('DT - Equal Opportunity')
                    df_dt_Eop = SingleModelParetoDT(df_dt_Eop, Eop_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_max_depth, df_dt, 'Equal opportunity')
                    df_dt_Eop.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eop[0] = df_dt_Eop
                    st.write(df_dt_Eop)
                    csv = convert_df(df_dt_Eop)
                    download_button_str = download_button(csv, "dt_pareto_Eop.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Acc_ok:
                    st.subheader('DT - Accuracy Equality')
                    df_dt_Acc = SingleModelParetoDT(df_dt_Acc, Acc_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_max_depth, df_dt, 'Accuracy equality')
                    df_dt_Acc.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Acc[0] = df_dt_Acc
                    st.write(df_dt_Acc)
                    csv = convert_df(df_dt_Acc)
                    download_button_str = download_button(csv, "dt_pareto_Acc.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eq1_ok:
                    st.subheader('DT - Equalized Odds')
                    df_dt_Eq1 = SingleModelParetoDT(df_dt_Eq1, Eq1_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_max_depth, df_dt, 'Equalized odds')
                    df_dt_Eq1.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eq1[0] = df_dt_Eq1
                    st.write(df_dt_Eq1)
                    csv = convert_df(df_dt_Eq1)
                    download_button_str = download_button(csv, "dt_pareto_Eq1.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Pty_ok:
                    st.subheader('DT - Statistical Parity')
                    df_dt_Pty = SingleModelParetoDT(df_dt_Pty, Pty_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_max_depth, df_dt, 'Statistical parity')
                    df_dt_Pty.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Pty[0] = df_dt_Pty
                    st.write(df_dt_Pty)
                    csv = convert_df(df_dt_Pty)
                    download_button_str = download_button(csv, "dt_pareto_Pty.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                st.subheader('DT - Fairness Metrics Comparison')

                if prp_ok:
                    prp_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_dt_prp['Accuracy'],
                        y=df_dt_prp['Predictive parity'],
                        name='Predictive parity',
                        marker=dict(size=6, color="red", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_dt_prp['min_samples_leaf'], df_dt_prp['min_samples_split'], df_dt_prp['max_features'], 
                                                    df_dt_prp['criterion'], df_dt_prp['class weight'], df_dt_prp['max_depth']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Predictive parity: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_dt_prp['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_dt_prp['Predictive parity SEM'], visible=True),
                        )
                else: prp_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if prE_ok:
                    prE_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_dt_prE['Accuracy'],
                            y=df_dt_prE['Predictive equality'],
                            name='Predictive equality',
                            marker=dict(size=6, color="green", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_dt_prE['min_samples_leaf'], df_dt_prE['min_samples_split'], df_dt_prE['max_features'], 
                                                    df_dt_prE['criterion'], df_dt_prE['class weight'], df_dt_prE['max_depth']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Predictive equality: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_dt_prE['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_dt_prE['Predictive equality SEM'], visible=True),
                        )
                else: prE_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Eop_ok:
                    Eop_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_dt_Eop['Accuracy'],
                            y=df_dt_Eop['Equal opportunity'],
                            name='Equal opportunity',
                            marker=dict(size=6, color="gold", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_dt_Eop['min_samples_leaf'], df_dt_Eop['min_samples_split'], df_dt_Eop['max_features'], 
                                                    df_dt_Eop['criterion'], df_dt_Eop['class weight'], df_dt_Eop['max_depth']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Equal opportunity: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_dt_Eop['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_dt_Eop['Equal opportunity SEM'], visible=True),
                        )
                else: Eop_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Acc_ok:
                    Acc_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_dt_Acc['Accuracy'],
                            y=df_dt_Acc['Accuracy equality'],
                            name='Accuracy equality',
                            marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_dt_Acc['min_samples_leaf'], df_dt_Acc['min_samples_split'], df_dt_Acc['max_features'], 
                                                    df_dt_Acc['criterion'], df_dt_Acc['class weight'], df_dt_Acc['max_depth']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Accuracy equality: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_dt_Acc['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_dt_Acc['Accuracy equality SEM'], visible=True),
                        )
                else: Acc_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Eq1_ok:
                    Eq1_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_dt_Eq1['Accuracy'],
                            y=df_dt_Eq1['Equalized odds'],
                            name='Equalized odds',
                            marker=dict(size=6, color="black", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_dt_Eq1['min_samples_leaf'], df_dt_Eq1['min_samples_split'], df_dt_Eq1['max_features'], 
                                                    df_dt_Eq1['criterion'], df_dt_Eq1['class weight'], df_dt_Eq1['max_depth']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Equalized odds: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_dt_Eq1['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_dt_Eq1['Equalized odds SEM'], visible=True),
                            )
                else: Eq1_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Pty_ok:
                    Pty_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_dt_Pty['Accuracy'],
                                y=df_dt_Pty['Statistical parity'],
                                name='Statistical parity',
                                marker=dict(size=6, color="blue", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_dt_Pty['min_samples_leaf'], df_dt_Pty['min_samples_split'], df_dt_Pty['max_features'], 
                                                    df_dt_Pty['criterion'], df_dt_Pty['class weight'], df_dt_Pty['max_depth']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'min_samples_leaf: %{customdata[0]} <br>' +
                                'min_samples_split: %{customdata[1]} <br>' +
                                'max_features: %{customdata[2]} <br>' +
                                'criterion: %{customdata[3]} <br>' +
                                'class_weight: %{customdata[4]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_dt_Pty['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_dt_Pty['Predictive parity SEM'], visible=True),
                    )
                else: Pty_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")

                data_legend = [prp_scatter, prE_scatter, Eop_scatter, Acc_scatter, Eq1_scatter, Pty_scatter]

                fig = go.Figure(data=data_legend)

                fig.update_yaxes(title='Fairness', title_font=dict(size=20), visible=True)
                fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
                fig.update_traces(line_shape="vh")
                fig.update_layout(yaxis = dict(tickfont = dict(size=14)), xaxis = dict(tickfont = dict(size=15)))

                st.plotly_chart(fig)

            if checkDT and opt_method=='MOBO':

                n_features = len(X.columns)

                for notion in notions2:

                    class MLP:
                        @property
                        def configspace(self) -> ConfigurationSpace:
                            cs = ConfigurationSpace()

                            criterion = Categorical("criterion", ["gini", "entropy", "log_loss"], default="gini")
                            max_depth = Integer("max_depth", (1, 100), default=30)
                            splitter = Categorical("splitter", ["random", "best"], default="best")
                            max_features = Integer("max_features", (1, n_features), default=n_features)
                            min_samples_split = Integer("min_samples_split", (2, 40), default=20)
                            min_samples_leafs = Integer("min_samples_leaf", (1, 40), default=20)

                            cs.add_hyperparameters([max_depth, criterion, max_features, min_samples_split, min_samples_leafs, splitter])
                            return cs

                        def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:

                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                
                                classifier = DecisionTreeClassifier(
                                    criterion=config["criterion"],
                                    max_features=config["max_features"],
                                    min_samples_split=config["min_samples_split"],
                                    min_samples_leaf=config["min_samples_leaf"],
                                    splitter=config["splitter"],
                                    max_depth=config["max_depth"],
                                    random_state=seed,
                                )
                                
                                test_scores = []
                                fair = []

                                for seed in range(floor(n_seeds/2)):
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
                                    X_privileged_test = X_test[(X_test['Privileged'] > 0)]
                                    X_not_privileged_test = X_test[(X_test['Privileged'] == 0)]
                                    y_privileged_test = y_test[(X_test['Privileged'] > 0)]
                                    y_not_privileged_test = y_test[(X_test['Privileged'] == 0)]
                                    classifier.fit(X_train, y_train)
                                    test_scores.append(classifier.score(X_test, y_test))
                                    fair.append(fairness(X_privileged_test, y_privileged_test, 
                                                    X_not_privileged_test, y_not_privileged_test, 
                                                    classifier, False))
                                    
                                fairness_metrics = np.array(fair)

                            return {
                                "accuracy": 1 - np.mean(test_scores),
                                notion: np.mean(fairness_metrics[:, notions.index(notion)]),
                                #"predictive equality": np.mean(fairness_metrics[:, 1]),
                                #"equal opportunity": np.mean(fairness_metrics[:, 2]),
                                #"accuracy equality": np.mean(fairness_metrics[:, 3]),
                                #"equaized odds": np.mean(fairness_metrics[:, 4]),
                                #"statistical parity": np.mean(fairness_metrics[:, 5]),
                            }

                    def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
                        """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
                        average_costs = []
                        average_pareto_costs = []
                        for config in smac.runhistory.get_configs():
                            # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
                            average_cost = smac.runhistory.average_cost(config)

                            if config in incumbents:
                                average_pareto_costs += [average_cost]
                            else:
                                average_costs += [average_cost]

                        # Let's work with a numpy array
                        costs = np.vstack(average_costs)
                        pareto_costs = np.vstack(average_pareto_costs)
                        pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

                        costs_x, costs_y = costs[:, 0], costs[:, 1]
                        pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]
                        
                        plot_x = 1-costs_x
                        plot_y = 1-costs_y
                        plot_x_pareto = 1-pareto_costs_x
                        plot_y_pareto = 1-pareto_costs_y

                        plt.scatter(plot_x, plot_y, marker="x", label="Configuration")
                        plt.scatter(plot_x_pareto, plot_y_pareto, marker="x", c="r", label="Incumbent")
                        plt.step(
                            [plot_x_pareto[0]] + plot_x_pareto.tolist() + [np.min(plot_x)],  # We add bounds
                            [np.min(plot_y)] + plot_y_pareto.tolist() + [np.max(plot_y_pareto)],  # We add bounds
                            where="post",
                            linestyle=":",
                        )

                        plt.title("Pareto-Front")
                        plt.xlabel(smac.scenario.objectives[0])
                        plt.ylabel(smac.scenario.objectives[1])
                        plt.legend()
                        plt.show()


                    if __name__ == "__main__":
                        mlp = MLP()
                        objectives = ["accuracy", notion]

                        # Define our environment variables
                        scenario = Scenario(
                            mlp.configspace,
                            objectives=objectives,
                            walltime_limit=60000,  # After 30 seconds, we stop the hyperparameter optimization
                            n_trials=100,  # Evaluate max 200 different trials
                            n_workers=1,
                        )
                        
                        initial_design = LatinHypercubeInitialDesign(scenario, n_configs_per_hyperparameter=6)

                        # We want to run five random configurations before starting the optimization.
                        #initial_design = HPOFacade.get_initial_design(scenario, n_configs_per_hyperparameter=6)
                        multi_objective_algorithm = ParEGO(scenario)
                        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

                        # Create our SMAC object and pass the scenario and the train method
                        smac = HPOFacade(
                            scenario,
                            mlp.train,
                            initial_design=initial_design,
                            multi_objective_algorithm=multi_objective_algorithm,
                            intensifier=intensifier,
                            overwrite=True,
                        )

                        # Let's optimize
                        incumbents = smac.optimize()

                        # Get cost of default configuration
                        default_cost = smac.validate(mlp.configspace.get_default_configuration())
                        print(f"Validated costs from default config: \n--- {default_cost}\n")

                        print("Validated costs from the Pareto front (incumbents):")
                        for incumbent in incumbents:
                            cost = smac.validate(incumbent)
                            print("---", cost)

                        # Let's plot a pareto front
                        plot_pareto(smac, incumbents)

                    criterion = []
                    max_depth = []
                    max_features = []
                    min_samples_leaf = []
                    min_samples_split = []
                    splitter = []
                    accuracy = []
                    prp = []

                    for i in range(len(incumbents)):
                        
                        criterion.append(incumbents[i]['criterion'])
                        max_depth.append(incumbents[i]['max_depth'])
                        max_features.append(incumbents[i]['max_features'])
                        min_samples_leaf.append(incumbents[i]['min_samples_leaf'])
                        min_samples_split.append(incumbents[i]['min_samples_split'])
                        splitter.append(incumbents[i]['splitter'])
                        accuracy.append(1-smac.validate(incumbents[i])[0])
                        prp.append(1-smac.validate(incumbents[i])[1])
                        
                    solutions_space = {
                        'accuracy': accuracy, 
                        notion: prp,
                        'criterion': criterion,
                        'max_depth': max_depth,
                        'splitter': splitter,
                        'max_features': max_features,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': min_samples_split,
                        }

                    dfp_dt = pd.DataFrame(data=solutions_space)
                    dfp_dt.sort_values('accuracy', inplace=True, ignore_index=True)
                    dfp_dt.replace(to_replace=[None], value='None', inplace=True)

                    if notion=='predictive parity':
                        df_MODEL_prp[0] = dfp_dt
                    if notion=='predictive equality':
                        df_MODEL_prE[0] = dfp_dt
                    if notion=='equal opportunity':
                        df_MODEL_Eop[0] = dfp_dt
                    if notion=='accuracy equality':
                        df_MODEL_Acc[0] = dfp_dt
                    if notion=='equalized odds':
                        df_MODEL_Eq1[0] = dfp_dt
                    if notion=='statistical parity':
                        df_MODEL_Pty[0] = dfp_dt

                    fig = px.line(dfp_dt, x = 'accuracy', y = notion,
                                    hover_data=['min_samples_leaf','min_samples_split','criterion','max_features','max_depth'])
                    fig.update_traces(mode="markers+lines", line_shape="vh", line_dash="dash")
                    fig.update_traces(marker=dict(line=dict(width=1.2, color='DarkSlateGrey')))

                    fig.update_yaxes(title_font=dict(size=20))
                    fig.update_xaxes(title_font=dict(size=20))

                    st.subheader('Decision Tree Classifier:' + notion)

                    st.plotly_chart(fig)

                    st.write(dfp_dt.head())

                    csv = convert_df(dfp_dt)

                    download_button_str = download_button(csv, "dtc_MOBO.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)

                    st.write('---')


            ###################################################
            ###################################################
            ###--------- SUPPORT VECTOR CLASSIFIER ---------###
            ###################################################
            ###################################################

            if checkSVC and opt_method=='Grid Search':

                test_scores = []
                test_scores_privileged = []
                test_scores_not_privileged = []

                mean_test_scores = []
                mean_test_scores_privileged = []
                mean_test_scores_not_privileged = []

                sem_test_scores = []

                fairness_metrics = []

                mean_prp = []
                mean_prE = []
                mean_Eop = []
                mean_Acc = []
                mean_Eq1 = []
                mean_Pty = []

                sem_prp = []
                sem_prE = []
                sem_Eop = []
                sem_Acc = []
                sem_Eq1 = []
                sem_Pty = []

                hyp_penalty = []
                hyp_C = []
                hyp_loss = []
                hyp_fit_intercept = []
                hyp_intercept_scaling = []

                output = []
                i = 0
                total_combinations = (len(Cs) * len(penalties) * len(fit_intercepts) * intercept_scalings.size * len(losses))*0.57

                progress_text = "Fitting Support Vector Classifiers..."
                my_bar = st.progress(0, text=progress_text)

                cond_losses = ['squared_hinge']
                cond_intercept_scalings = [1]

                for penalty in penalties :
                    if penalty=='l1': losses2=cond_losses
                    else: losses2=losses
                    for c in Cs :
                        for loss in losses2:
                            for fit_intercept in fit_intercepts:
                                if fit_intercept==True: intercept_scalings2=intercept_scalings
                                else: intercept_scalings2=cond_intercept_scalings
                                for intercept_scaling in intercept_scalings2:
                                    if penalty=='l2' and loss=='hinge': dual=True
                                    else: dual=False
                                    for seed in range(n_seeds) :
                                                    
                                        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=seed, train_size=split_size)
                                        X_privileged_test = X_test[(X_test[X_sen] > 0)]
                                        X_not_privileged_test = X_test[(X_test[X_sen] == 0)]
                                        y_privileged_test = y_test[(X_test[X_sen] > 0)]
                                        y_not_privileged_test = y_test[(X_test[X_sen] == 0)]
                                        svc = LinearSVC(penalty=penalty, C=c, loss=loss, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, dual=dual, random_state=0)
                                        svc.fit(X_train, y_train)
                                        test_scores.append(svc.score(X_test, y_test))
                                        test_scores_privileged.append(svc.score(X_privileged_test, y_privileged_test))
                                        test_scores_not_privileged.append(svc.score(X_not_privileged_test, y_not_privileged_test))

                                        output.append(fc.fairness(X_privileged_test, y_privileged_test, X_not_privileged_test, y_not_privileged_test, svc, False))

                                    fairness_metrics = np.array(output)

                                    ## Fairness metrics
                                    mean_prp.append(fairness_metrics[:, 0].mean())
                                    mean_prE.append(fairness_metrics[:, 1].mean())
                                    mean_Eop.append(fairness_metrics[:, 2].mean())
                                    mean_Acc.append(fairness_metrics[:, 3].mean())
                                    mean_Eq1.append(fairness_metrics[:, 4].mean())
                                    mean_Pty.append(fairness_metrics[:, 5].mean())

                                    sem_prp.append(sem(fairness_metrics[:, 0]))
                                    sem_prE.append(sem(fairness_metrics[:, 1]))
                                    sem_Eop.append(sem(fairness_metrics[:, 2]))
                                    sem_Acc.append(sem(fairness_metrics[:, 3]))
                                    sem_Eq1.append(sem(fairness_metrics[:, 4]))
                                    sem_Pty.append(sem(fairness_metrics[:, 5]))

                                    ## Accuracy scores
                                    mean_test_scores.append(mean(test_scores))
                                    mean_test_scores_privileged.append(mean(test_scores_privileged))
                                    mean_test_scores_not_privileged.append(mean(test_scores_not_privileged))

                                    sem_test_scores.append(sem(test_scores))

                                    ## Hyperparameters
                                    hyp_penalty.append(penalty)
                                    hyp_C.append(c)
                                    hyp_loss.append(loss)
                                    hyp_fit_intercept.append(fit_intercept)
                                    hyp_intercept_scaling.append(intercept_scaling)

                                    ## Reset for each hyperparameter combination
                                    output = []
                                    test_scores = []
                                    test_scores_privileged = []
                                    test_scores_not_privileged = []
                                    i = i + 1
                                    my_bar.progress(i/total_combinations, text=progress_text)
                
                my_bar.progress(0.999, text=progress_text)                
                st.subheader('Support Vector Classifier: Results')

                solutions_space = {
                    'penalty': hyp_penalty, 
                    'C': hyp_C,
                    'loss': hyp_loss,
                    'fit_intercept': hyp_fit_intercept,
                    'intercept_scaling': hyp_intercept_scaling,
                    'Accuracy': mean_test_scores,
                    'Accuracy std': sem_test_scores,
                    'Predictive parity': mean_prp,
                    'Predictive parity SEM': sem_prp,
                    'Predictive equality': mean_prE,
                    'Predictive equality SEM': sem_prE,
                    'Equal opportunity': mean_Eop,
                    'Equal opportunity SEM': sem_Eop,
                    'Accuracy equality': mean_Acc,
                    'Accuracy equality SEM': sem_Acc,
                    'Equalized odds': mean_Eq1,
                    'Equalized odds SEM': sem_Eq1,
                    'Statistical parity': mean_Pty,
                    'Statistical parity SEM': sem_Pty,
                }

                df_svc = pd.DataFrame(data=solutions_space)
                df_svc.replace(to_replace=[None], value='None', inplace=True)
                df_MODEL[1] = df_svc
                st.write(df_svc.head(10))

                csv = convert_df(df_svc)

                download_button_str = download_button(csv, "svc_full_table.csv", 'Download Table', pickle_it=False)
                st.markdown(download_button_str, unsafe_allow_html=True)

                st.write('---')

                if prp_ok:
                    df_svc_prp = pt.pareto_frontier(Xs=df_svc['Accuracy'], Ys=df_svc['Predictive parity'], name='Predictive parity', maxX=True, maxY=True)
                if prE_ok:
                    df_svc_prE = pt.pareto_frontier(Xs=df_svc['Accuracy'], Ys=df_svc['Predictive equality'], name='Predictive equality', maxX=True, maxY=True)
                if Eop_ok:
                    df_svc_Eop = pt.pareto_frontier(Xs=df_svc['Accuracy'], Ys=df_svc['Equal opportunity'], name='Equal opportunity', maxX=True, maxY=True)
                if Acc_ok:    
                    df_svc_Acc = pt.pareto_frontier(Xs=df_svc['Accuracy'], Ys=df_svc['Accuracy equality'], name='Accuracy equality', maxX=True, maxY=True)
                if Eq1_ok:
                    df_svc_Eq1 = pt.pareto_frontier(Xs=df_svc['Accuracy'], Ys=df_svc['Equalized odds'], name='Equalized odds', maxX=True, maxY=True)
                if Pty_ok:
                    df_svc_Pty = pt.pareto_frontier(Xs=df_svc['Accuracy'], Ys=df_svc['Statistical parity'], name='Statistical parity', maxX=True, maxY=True)

                mean_test_scores_np = np.array(mean_test_scores)
                sem_test_scores_np = np.array(sem_test_scores)
                prp_np = np.array(mean_prp)
                prE_np = np.array(mean_prE)
                Eop_np = np.array(mean_Eop)
                Acc_np = np.array(mean_Acc)
                Eq1_np = np.array(mean_Eq1)
                Pty_np = np.array(mean_Pty)

                sem_prp_np = np.array(sem_prp)
                sem_prE_np = np.array(sem_prE)
                sem_Eop_np = np.array(sem_Eop)
                sem_Acc_np = np.array(sem_Acc)
                sem_Eq1_np = np.array(sem_Eq1)
                sem_Pty_np = np.array(sem_Pty)

                if prp_ok:
                    st.subheader('SVC - Predictive Parity')
                    df_svc_prp = SingleModelParetoSVC(df_svc_prp, prp_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, hyp_loss, hyp_fit_intercept, hyp_intercept_scaling, df_svc, 'Predictive parity')
                    df_svc_prp.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prp[1] = df_svc_prp
                    st.write(df_svc_prp)
                    csv = convert_df(df_svc_prp)

                    download_button_str = download_button(csv, "svc_pareto_prp.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if prE_ok:
                    st.subheader('SVC - Predictive Equality')
                    df_svc_prE = SingleModelParetoSVC(df_svc_prE, prE_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, hyp_loss, hyp_fit_intercept, hyp_intercept_scaling, df_svc, 'Predictive equality')
                    df_svc_prE.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prE[1] = df_svc_prE
                    st.write(df_svc_prE)
                    csv = convert_df(df_svc_prE)
                    download_button_str = download_button(csv, "svc_pareto_prE.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eop_ok:
                    st.subheader('SVC - Equal Opportunity')
                    df_svc_Eop = SingleModelParetoSVC(df_svc_Eop, Eop_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, hyp_loss, hyp_fit_intercept, hyp_intercept_scaling, df_svc, 'Equal opportunity')
                    df_svc_Eop.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eop[1] = df_svc_Eop
                    st.write(df_svc_Eop)
                    csv = convert_df(df_svc_Eop)
                    download_button_str = download_button(csv, "svc_pareto_Eop.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Acc_ok:
                    st.subheader('SVC - Accuracy Equality')
                    df_svc_Acc = SingleModelParetoSVC(df_svc_Acc, Acc_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, hyp_loss, hyp_fit_intercept, hyp_intercept_scaling, df_svc, 'Accuracy equality')
                    df_svc_Acc.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Acc[1] = df_svc_Acc
                    st.write(df_svc_Acc)
                    csv = convert_df(df_svc_Acc)
                    download_button_str = download_button(csv, "svc_pareto_Acc.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eq1_ok:
                    st.subheader('SVC - Equalized Odds')
                    df_svc_Eq1 = SingleModelParetoSVC(df_svc_Eq1, Eq1_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, hyp_loss, hyp_fit_intercept, hyp_intercept_scaling, df_svc, 'Equalized odds')
                    df_svc_Eq1.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eq1[1] = df_svc_Eq1
                    st.write(df_svc_Eq1)
                    csv = convert_df(df_svc_Eq1)
                    download_button_str = download_button(csv, "svc_pareto_Eq1.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Pty_ok:
                    st.subheader('SVC - Statistical Parity')
                    df_svc_Pty = SingleModelParetoSVC(df_svc_Pty, Pty_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, hyp_loss, hyp_fit_intercept, hyp_intercept_scaling, df_svc, 'Statistical parity')
                    df_svc_Pty.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Pty[1] = df_svc_Pty
                    st.write(df_svc_Pty)
                    csv = convert_df(df_svc_Pty)
                    download_button_str = download_button(csv, "svc_pareto_Pty.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                st.subheader('SVC - Fairness Metrics Comparison')

                if prp_ok:
                    prp_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_svc_prp['Accuracy'],
                        y=df_svc_prp['Predictive parity'],
                        name='Predictive parity',
                        marker=dict(size=6, color="red", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_svc_prp['penalty'], df_svc_prp['C'], df_svc_prp['loss'], df_svc_prp['fit_intercept'], df_svc_prp['intercept_scaling']), axis=-1),
                        hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                        'Predictive parity: %{y:,.4f} <br>' + 
                        'penalty: %{customdata[0]} <br>' +
                        'C: %{customdata[1]} <br>' +
                        'loss: %{customdata[2]} <br>' +
                        'fit_intercept: %{customdata[3]} <br>' +
                        'intercept_scaling: %{customdata[4]} <br>' +
                        '<extra></extra>'),
                        error_x = dict(type='data', array=df_svc_prp['Accuracy SEM'], visible=True),
                        error_y = dict(type='data', array=df_svc_prp['Predictive parity SEM'], visible=True),
                        )
                else: prp_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if prE_ok:
                    prE_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_svc_prE['Accuracy'],
                        y=df_svc_prE['Predictive equality'],
                        name='Predictive equality',
                        marker=dict(size=6, color="green", symbol='circle',line=dict(width=2,
                                                    color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_svc_prE['penalty'], df_svc_prE['C'], df_svc_prE['loss'], df_svc_prE['fit_intercept'], df_svc_prE['intercept_scaling']), axis=-1),
                        hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                        'Predictive parity: %{y:,.4f} <br>' + 
                        'penalty: %{customdata[0]} <br>' +
                        'C: %{customdata[1]} <br>' +
                        'loss: %{customdata[2]} <br>' +
                        'fit_intercept: %{customdata[3]} <br>' +
                        'intercept_scaling: %{customdata[4]} <br>' +
                        '<extra></extra>'),
                        error_x = dict(type='data', array=df_svc_prE['Accuracy SEM'], visible=True),
                        error_y = dict(type='data', array=df_svc_prE['Predictive equality SEM'], visible=True),
                        )
                else: prE_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Eop_ok:
                    Eop_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_svc_Eop['Accuracy'],
                        y=df_svc_Eop['Equal opportunity'],
                        name='Equal opportunity',
                        marker=dict(size=6, color="gold", symbol='circle',line=dict(width=2,
                                                    color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_svc_Eop['penalty'], df_svc_Eop['C'], df_svc_Eop['loss'], df_svc_Eop['fit_intercept'], df_svc_Eop['intercept_scaling']), axis=-1),
                        hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                        'Predictive parity: %{y:,.4f} <br>' + 
                        'penalty: %{customdata[0]} <br>' +
                        'C: %{customdata[1]} <br>' +
                        'loss: %{customdata[2]} <br>' +
                        'fit_intercept: %{customdata[3]} <br>' +
                        'intercept_scaling: %{customdata[4]} <br>' +
                        '<extra></extra>'),
                        error_x = dict(type='data', array=df_svc_Eop['Accuracy SEM'], visible=True),
                        error_y = dict(type='data', array=df_svc_Eop['Equal opportunity SEM'], visible=True),
                    )
                else: Eop_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Acc_ok:
                    Acc_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_svc_Acc['Accuracy'],
                        y=df_svc_Acc['Accuracy equality'],
                        name='Accuracy equality',
                        marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=2,
                                                    color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_svc_Acc['penalty'], df_svc_Acc['C'], df_svc_Acc['loss'], df_svc_Acc['fit_intercept'], df_svc_Acc['intercept_scaling']), axis=-1),
                        hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                        'Predictive parity: %{y:,.4f} <br>' + 
                        'penalty: %{customdata[0]} <br>' +
                        'C: %{customdata[1]} <br>' +
                        'loss: %{customdata[2]} <br>' +
                        'fit_intercept: %{customdata[3]} <br>' +
                        'intercept_scaling: %{customdata[4]} <br>' +
                        '<extra></extra>'),
                        error_x = dict(type='data', array=df_svc_Acc['Accuracy SEM'], visible=True),
                        error_y = dict(type='data', array=df_svc_Acc['Accuracy equality SEM'], visible=True),
                    )
                else: Acc_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Eq1_ok:
                    Eq1_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_svc_Eq1['Accuracy'],
                        y=df_svc_Eq1['Equalized odds'],
                        name='Equalized odds',
                        marker=dict(size=6, color="black", symbol='circle',line=dict(width=2,
                                                    color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_svc_Eq1['penalty'], df_svc_Eq1['C'], df_svc_Eq1['loss'], df_svc_Eq1['fit_intercept'], df_svc_Eq1['intercept_scaling']), axis=-1),
                        hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                        'Predictive parity: %{y:,.4f} <br>' + 
                        'penalty: %{customdata[0]} <br>' +
                        'C: %{customdata[1]} <br>' +
                        'loss: %{customdata[2]} <br>' +
                        'fit_intercept: %{customdata[3]} <br>' +
                        'intercept_scaling: %{customdata[4]} <br>' +
                        '<extra></extra>'),
                        error_x = dict(type='data', array=df_svc_Eq1['Accuracy SEM'], visible=True),
                        error_y = dict(type='data', array=df_svc_Eq1['Equalized odds SEM'], visible=True),
                        )
                else: Eq1_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Pty_ok:
                    Pty_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_svc_Pty['Accuracy'],
                        y=df_svc_Pty['Statistical parity'],
                        name='Statistical parity',
                        marker=dict(size=6, color="blue", symbol='circle',line=dict(width=2,
                                                    color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_svc_Pty['penalty'], df_svc_Pty['C'], df_svc_Pty['loss'], df_svc_Pty['fit_intercept'], df_svc_Pty['intercept_scaling']), axis=-1),
                        hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                        'Predictive parity: %{y:,.4f} <br>' + 
                        'penalty: %{customdata[0]} <br>' +
                        'C: %{customdata[1]} <br>' +
                        'loss: %{customdata[2]} <br>' +
                        'fit_intercept: %{customdata[3]} <br>' +
                        'intercept_scaling: %{customdata[4]} <br>' +
                        '<extra></extra>'),
                        error_x = dict(type='data', array=df_svc_Pty['Accuracy SEM'], visible=True),
                        error_y = dict(type='data', array=df_svc_Pty['Predictive parity SEM'], visible=True),
                        )
                else: Pty_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")

                data_legend = [prp_scatter, prE_scatter, Eop_scatter, Acc_scatter, Eq1_scatter, Pty_scatter]

                fig = go.Figure(data=data_legend)

                fig.update_yaxes(title='Fairness', title_font=dict(size=20), visible=True)
                fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
                fig.update_traces(line_shape="vh")
                fig.update_layout(yaxis = dict(tickfont = dict(size=14)), xaxis = dict(tickfont = dict(size=15)))

                st.plotly_chart(fig)

            if checkSVC and opt_method=='MOBO':

                n_features = len(X.columns)

                for notion in notions2:

                    class MLP:
                        @property
                        def configspace(self) -> ConfigurationSpace:
                            cs = ConfigurationSpace()

                            penalty = Categorical("penalty", ['l1', 'l2'], default="l1")
                            C = Float("C", (0.001, 1000.0), default=1.0, log=True)
                            loss = Categorical("loss", ['hinge', 'squared_hinge'], default='hinge')
                            fit_intercept = Categorical("fit_intercept", [True, False], default=True)
                            intercept_scaling = Float("intercept_scaling", (0.1, 10.0), default=1.0)

                            use_loss = InCondition(child=loss, parent=penalty, values=["l2"])
                            use_intercept_scaling = InCondition(child=intercept_scaling, parent=fit_intercept, values=[True])

                            cs.add_hyperparameters([penalty, C, loss, fit_intercept, intercept_scaling])
                            cs.add_conditions([use_loss, use_intercept_scaling])

                            return cs

                        def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:

                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                
                                classifier = LinearSVC(penalty=config['penalty'], 
                                                C=config['C'], 
                                                loss=config['loss'], 
                                                fit_intercept=config['fit_intercept'],
                                                intercept_scaling=config['intercept_scaling'],
                                                dual=False, 
                                                random_state=seed)
                                
                                test_scores = []
                                fair = []

                                for seed in range(floor(n_seeds/2)):
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
                                    X_privileged_test = X_test[(X_test['Privileged'] > 0)]
                                    X_not_privileged_test = X_test[(X_test['Privileged'] == 0)]
                                    y_privileged_test = y_test[(X_test['Privileged'] > 0)]
                                    y_not_privileged_test = y_test[(X_test['Privileged'] == 0)]
                                    classifier.fit(X_train, y_train)
                                    test_scores.append(classifier.score(X_test, y_test))
                                    fair.append(fairness(X_privileged_test, y_privileged_test, 
                                                    X_not_privileged_test, y_not_privileged_test, 
                                                    classifier, False))
                                    
                                fairness_metrics = np.array(fair)

                            return {
                                "accuracy": 1 - np.mean(test_scores),
                                notion: np.mean(fairness_metrics[:, notions.index(notion)]),
                            }

                    def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
                        """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
                        average_costs = []
                        average_pareto_costs = []
                        for config in smac.runhistory.get_configs():
                            # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
                            average_cost = smac.runhistory.average_cost(config)

                            if config in incumbents:
                                average_pareto_costs += [average_cost]
                            else:
                                average_costs += [average_cost]

                        # Let's work with a numpy array
                        costs = np.vstack(average_costs)
                        pareto_costs = np.vstack(average_pareto_costs)
                        pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

                        costs_x, costs_y = costs[:, 0], costs[:, 1]
                        pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]
                        
                        plot_x = 1-costs_x
                        plot_y = 1-costs_y
                        plot_x_pareto = 1-pareto_costs_x
                        plot_y_pareto = 1-pareto_costs_y

                        plt.scatter(plot_x, plot_y, marker="x", label="Configuration")
                        plt.scatter(plot_x_pareto, plot_y_pareto, marker="x", c="r", label="Incumbent")
                        plt.step(
                            [plot_x_pareto[0]] + plot_x_pareto.tolist() + [np.min(plot_x)],  # We add bounds
                            [np.min(plot_y)] + plot_y_pareto.tolist() + [np.max(plot_y_pareto)],  # We add bounds
                            where="post",
                            linestyle=":",
                        )

                        plt.title("Pareto-Front")
                        plt.xlabel(smac.scenario.objectives[0])
                        plt.ylabel(smac.scenario.objectives[1])
                        plt.legend()
                        plt.show()


                    if __name__ == "__main__":
                        mlp = MLP()
                        objectives = ["accuracy", notion]

                        # Define our environment variables
                        scenario = Scenario(
                            mlp.configspace,
                            objectives=objectives,
                            walltime_limit=60000,  # After 30 seconds, we stop the hyperparameter optimization
                            n_trials=100,  # Evaluate max 200 different trials
                            n_workers=1,
                        )
                        
                        initial_design = LatinHypercubeInitialDesign(scenario, n_configs_per_hyperparameter=6)

                        # We want to run five random configurations before starting the optimization.
                        #initial_design = HPOFacade.get_initial_design(scenario, n_configs_per_hyperparameter=6)
                        multi_objective_algorithm = ParEGO(scenario)
                        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

                        # Create our SMAC object and pass the scenario and the train method
                        smac = HPOFacade(
                            scenario,
                            mlp.train,
                            initial_design=initial_design,
                            multi_objective_algorithm=multi_objective_algorithm,
                            intensifier=intensifier,
                            overwrite=True,
                        )

                        # Let's optimize
                        incumbents = smac.optimize()

                        # Get cost of default configuration
                        default_cost = smac.validate(mlp.configspace.get_default_configuration())
                        print(f"Validated costs from default config: \n--- {default_cost}\n")

                        print("Validated costs from the Pareto front (incumbents):")
                        for incumbent in incumbents:
                            cost = smac.validate(incumbent)
                            print("---", cost)

                        # Let's plot a pareto front
                        plot_pareto(smac, incumbents)

                    penalty = []
                    C = []
                    loss = []
                    fit_intercept = []
                    intercept_scaling = []
                    accuracy = []
                    prp = []

                    for i in range(len(incumbents)):
                        
                        accuracy.append(1-smac.validate(incumbents[i])[0])
                        prp.append(1-smac.validate(incumbents[i])[1])
                        penalty.append(incumbents[i]['penalty'])
                        C.append(incumbents[i]['C'])
                        loss.append(incumbents[i]['loss'])
                        fit_intercept.append(incumbents[i]['fit_intercept'])
                        intercept_scaling.append(incumbents[i]['intercept_scaling'])
                        
                    solutions_space = {
                        'accuracy': accuracy, 
                        notion: prp,
                        'penalty': penalty,
                        'C': C,
                        'loss': loss,
                        'fit_intercept': fit_intercept,
                        'intercept_scaling': intercept_scaling,
                        }

                    dfp_svc = pd.DataFrame(data=solutions_space)
                    dfp_svc.sort_values('accuracy', inplace=True, ignore_index=True)
                    dfp_svc.replace(to_replace=[None], value='None', inplace=True)

                    if notion=='predictive parity':
                        df_MODEL_prp[1] = dfp_svc
                    if notion=='predictive equality':
                        df_MODEL_prE[1] = dfp_svc
                    if notion=='equal opportunity':
                        df_MODEL_Eop[1] = dfp_svc
                    if notion=='accuracy equality':
                        df_MODEL_Acc[1] = dfp_svc
                    if notion=='equalized odds':
                        df_MODEL_Eq1[1] = dfp_svc
                    if notion=='statistical parity':
                        df_MODEL_Pty[1] = dfp_svc

                    fig = px.line(dfp_svc, x = 'accuracy', y = notion,
                                    hover_data=['penalty','C','loss','fit_intercept','intercept_scaling'])
                    fig.update_traces(mode="markers+lines", line_shape="vh", line_dash="dash")
                    fig.update_traces(marker=dict(line=dict(width=1.2, color='DarkSlateGrey')))

                    fig.update_yaxes(title_font=dict(size=20))
                    fig.update_xaxes(title_font=dict(size=20))

                    st.subheader('Support Vector Classifier:' + notion)

                    st.plotly_chart(fig)

                    st.write(dfp_svc.head())

                    csv = convert_df(dfp_svc)

                    download_button_str = download_button(csv, "svc_MOBO.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)

                    st.write('---')

            #############################################
            #############################################
            ###--------- LOGISTIC REGRESSION ---------###
            #############################################
            #############################################

            if checkLR and opt_method == "Grid Search":

                test_scores = []
                test_scores_privileged = []
                test_scores_not_privileged = []

                mean_test_scores = []
                mean_test_scores_privileged = []
                mean_test_scores_not_privileged = []

                sem_test_scores = []

                fairness_metrics = []

                mean_prp = []
                mean_prE = []
                mean_Eop = []
                mean_Acc = []
                mean_Eq1 = []
                mean_Pty = []

                sem_prp = []
                sem_prE = []
                sem_Eop = []
                sem_Acc = []
                sem_Eq1 = []
                sem_Pty = []

                hyp_C = []
                hyp_penalty = []

                output = []
                i = 0
                total_combinations = len(LR_Cs) * len(LR_penalties)

                progress_text = "Fitting Logistic Curves..."
                my_bar = st.progress(0, text=progress_text)

                for penalty in LR_penalties :
                    for c in LR_Cs :
                        for seed in range(n_seeds) :
                                        
                            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=seed, train_size=split_size)
                            X_privileged_test = X_test[(X_test[X_sen] > 0)]
                            X_not_privileged_test = X_test[(X_test[X_sen] == 0)]
                            y_privileged_test = y_test[(X_test[X_sen] > 0)]
                            y_not_privileged_test = y_test[(X_test[X_sen] == 0)]
                            lr = LogisticRegression(penalty=penalty, C=c, random_state=0)
                            lr.fit(X_train, y_train)
                            test_scores.append(lr.score(X_test, y_test))
                            test_scores_privileged.append(lr.score(X_privileged_test, y_privileged_test))
                            test_scores_not_privileged.append(lr.score(X_not_privileged_test, y_not_privileged_test))

                            output.append(fc.fairness(X_privileged_test, y_privileged_test, X_not_privileged_test, y_not_privileged_test, lr, False))

                        fairness_metrics = np.array(output)

                        ## Fairness metrics
                        mean_prp.append(fairness_metrics[:, 0].mean())
                        mean_prE.append(fairness_metrics[:, 1].mean())
                        mean_Eop.append(fairness_metrics[:, 2].mean())
                        mean_Acc.append(fairness_metrics[:, 3].mean())
                        mean_Eq1.append(fairness_metrics[:, 4].mean())
                        mean_Pty.append(fairness_metrics[:, 5].mean())

                        sem_prp.append(sem(fairness_metrics[:, 0]))
                        sem_prE.append(sem(fairness_metrics[:, 1]))
                        sem_Eop.append(sem(fairness_metrics[:, 2]))
                        sem_Acc.append(sem(fairness_metrics[:, 3]))
                        sem_Eq1.append(sem(fairness_metrics[:, 4]))
                        sem_Pty.append(sem(fairness_metrics[:, 5]))

                        ## Accuracy scores
                        mean_test_scores.append(mean(test_scores))
                        mean_test_scores_privileged.append(mean(test_scores_privileged))
                        mean_test_scores_not_privileged.append(mean(test_scores_not_privileged))

                        sem_test_scores.append(sem(test_scores))

                        ## Hyperparameters
                        hyp_C.append(c)
                        hyp_penalty.append(penalty)

                        ## Reset for each hyperparameter combination
                        output = []
                        test_scores = []
                        test_scores_privileged = []
                        test_scores_not_privileged = []
                        i = i + 1
                        my_bar.progress(i/total_combinations, text=progress_text)
                                
                st.subheader('Logistic Regression: Results')

                solutions_space = {
                    'penalty': hyp_penalty, 
                    'C': hyp_C,
                    'Accuracy': mean_test_scores,
                    'Accuracy std': sem_test_scores,
                    'Predictive parity': mean_prp,
                    'Predictive parity SEM': sem_prp,
                    'Predictive equality': mean_prE,
                    'Predictive equality SEM': sem_prE,
                    'Equal opportunity': mean_Eop,
                    'Equal opportunity SEM': sem_Eop,
                    'Accuracy equality': mean_Acc,
                    'Accuracy equality SEM': sem_Acc,
                    'Equalized odds': mean_Eq1,
                    'Equalized odds SEM': sem_Eq1,
                    'Statistical parity': mean_Pty,
                    'Statistical parity SEM': sem_Pty,
                }

                df_lr = pd.DataFrame(data=solutions_space)
                df_lr.replace(to_replace=[None], value='None', inplace=True)
                df_MODEL[2] = df_lr
                st.write(df_lr.head(10))

                csv = convert_df(df_lr)

                download_button_str = download_button(csv, "lr_full_table.csv", 'Download Table', pickle_it=False)
                st.markdown(download_button_str, unsafe_allow_html=True)

                st.write('---')

                if prp_ok:
                    df_lr_prp = pt.pareto_frontier(Xs=df_lr['Accuracy'], Ys=df_lr['Predictive parity'], name='Predictive parity', maxX=True, maxY=True)
                if prE_ok:
                    df_lr_prE = pt.pareto_frontier(Xs=df_lr['Accuracy'], Ys=df_lr['Predictive equality'], name='Predictive equality', maxX=True, maxY=True)
                if Eop_ok:
                    df_lr_Eop = pt.pareto_frontier(Xs=df_lr['Accuracy'], Ys=df_lr['Equal opportunity'], name='Equal opportunity', maxX=True, maxY=True)
                if Acc_ok:    
                    df_lr_Acc = pt.pareto_frontier(Xs=df_lr['Accuracy'], Ys=df_lr['Accuracy equality'], name='Accuracy equality', maxX=True, maxY=True)
                if Eq1_ok:
                    df_lr_Eq1 = pt.pareto_frontier(Xs=df_lr['Accuracy'], Ys=df_lr['Equalized odds'], name='Equalized odds', maxX=True, maxY=True)
                if Pty_ok:
                    df_lr_Pty = pt.pareto_frontier(Xs=df_lr['Accuracy'], Ys=df_lr['Statistical parity'], name='Statistical parity', maxX=True, maxY=True)

                mean_test_scores_np = np.array(mean_test_scores)
                sem_test_scores_np = np.array(sem_test_scores)
                prp_np = np.array(mean_prp)
                prE_np = np.array(mean_prE)
                Eop_np = np.array(mean_Eop)
                Acc_np = np.array(mean_Acc)
                Eq1_np = np.array(mean_Eq1)
                Pty_np = np.array(mean_Pty)

                sem_prp_np = np.array(sem_prp)
                sem_prE_np = np.array(sem_prE)
                sem_Eop_np = np.array(sem_Eop)
                sem_Acc_np = np.array(sem_Acc)
                sem_Eq1_np = np.array(sem_Eq1)
                sem_Pty_np = np.array(sem_Pty)

                if prp_ok:
                    st.subheader('LR - Predictive Parity')
                    df_lr_prp = SingleModelParetoLR(df_lr_prp, prp_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, df_lr, 'Predictive parity')
                    df_lr_prp.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prp[2] = df_lr_prp
                    st.write(df_lr_prp)
                    csv = convert_df(df_lr_prp)

                    download_button_str = download_button(csv, "lr_pareto_prp.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if prE_ok:
                    st.subheader('LR - Predictive Equality')
                    df_lr_prE = SingleModelParetoLR(df_lr_prE, prE_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, df_lr, 'Predictive equality')
                    df_lr_prE.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prE[2] = df_lr_prE
                    st.write(df_lr_prE)
                    csv = convert_df(df_lr_prE)
                    download_button_str = download_button(csv, "lr_pareto_prE.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eop_ok:
                    st.subheader('LR - Equal Opportunity')
                    df_lr_Eop = SingleModelParetoLR(df_lr_Eop, Eop_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, df_lr, 'Equal opportunity')
                    df_lr_Eop.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eop[2] = df_lr_Eop
                    st.write(df_lr_Eop)
                    csv = convert_df(df_lr_Eop)
                    download_button_str = download_button(csv, "lr_pareto_Eop.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Acc_ok:
                    st.subheader('LR - Accuracy Equality')
                    df_lr_Acc = SingleModelParetoLR(df_lr_Acc, Acc_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, df_lr, 'Accuracy equality')
                    df_lr_Acc.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Acc[2] = df_lr_Acc
                    st.write(df_lr_Acc)
                    csv = convert_df(df_lr_Acc)
                    download_button_str = download_button(csv, "lr_pareto_Acc.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eq1_ok:
                    st.subheader('LR - Equalized Odds')
                    df_lr_Eq1 = SingleModelParetoLR(df_lr_Eq1, Eq1_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, df_lr, 'Equalized odds')
                    df_lr_Eq1.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eq1[3] = df_lr_Eq1
                    st.write(df_lr_Eq1)
                    csv = convert_df(df_lr_Eq1)
                    download_button_str = download_button(csv, "lr_pareto_Eq1.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Pty_ok:
                    st.subheader('LR - Statistical Parity')
                    df_lr_Pty = SingleModelParetoLR(df_lr_Pty, Pty_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_penalty, hyp_C, df_lr, 'Statistical parity')
                    df_lr_Pty.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Pty[4] = df_lr_Pty
                    st.write(df_lr_Pty)
                    csv = convert_df(df_lr_Pty)
                    download_button_str = download_button(csv, "lr_pareto_Pty.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                st.subheader('LR - Fairness Metrics Comparison')

                if prp_ok:
                    prp_scatter = go.Scatter(
                        mode='lines+markers',
                        x=df_lr_prp['Accuracy'],
                        y=df_lr_prp['Predictive parity'],
                        name='Predictive parity',
                        marker=dict(size=6, color="red", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                        line=dict(width=2),
                        customdata = np.stack((df_lr_prp['penalty'], df_lr_prp['C']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Predictive parity: %{y:,.4f} <br>' + 
                            'penalty: %{customdata[0]} <br>' +
                            'C: %{customdata[1]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_lr_prp['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_lr_prp['Predictive parity SEM'], visible=True),
                        )
                else: prp_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if prE_ok:
                    prE_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_lr_prE['Accuracy'],
                            y=df_lr_prE['Predictive equality'],
                            name='Predictive equality',
                            marker=dict(size=6, color="green", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_lr_prE['penalty'], df_lr_prE['C']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Predictive equality: %{y:,.4f} <br>' + 
                            'penalty: %{customdata[0]} <br>' +
                            'C: %{customdata[1]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_lr_prE['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_lr_prE['Predictive equality SEM'], visible=True),
                        )
                else: prE_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Eop_ok:
                    Eop_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_lr_Eop['Accuracy'],
                            y=df_lr_Eop['Equal opportunity'],
                            name='Equal opportunity',
                            marker=dict(size=6, color="gold", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_lr_Eop['penalty'], df_lr_Eop['C']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Equal opportunity: %{y:,.4f} <br>' + 
                            'penalty: %{customdata[0]} <br>' +
                            'C: %{customdata[1]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_lr_Eop['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_lr_Eop['Equal opportunity SEM'], visible=True),
                        )
                else: Eop_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Acc_ok:
                    Acc_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_lr_Acc['Accuracy'],
                            y=df_lr_Acc['Accuracy equality'],
                            name='Accuracy equality',
                            marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_lr_Acc['penalty'], df_lr_Acc['C']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Accuracy equality: %{y:,.4f} <br>' + 
                            'penalty: %{customdata[0]} <br>' +
                            'C: %{customdata[1]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_lr_Acc['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_lr_Acc['Accuracy equality SEM'], visible=True),
                        )
                else: Acc_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Eq1_ok:
                    Eq1_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_lr_Eq1['Accuracy'],
                            y=df_lr_Eq1['Equalized odds'],
                            name='Equalized odds',
                            marker=dict(size=6, color="black", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_lr_Eq1['penalty'], df_lr_Eq1['C']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Equalized odds: %{y:,.4f} <br>' + 
                            'penalty: %{customdata[0]} <br>' +
                            'C: %{customdata[1]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_lr_Eq1['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_lr_Eq1['Equalized odds SEM'], visible=True),
                            )
                else: Eq1_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Pty_ok:
                    Pty_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_lr_Pty['Accuracy'],
                                y=df_lr_Pty['Statistical parity'],
                                name='Statistical parity',
                                marker=dict(size=6, color="blue", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_lr_Pty['penalty'], df_lr_Pty['C']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'penalty: %{customdata[0]} <br>' +
                                'C: %{customdata[1]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_lr_Pty['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_lr_Pty['Predictive parity SEM'], visible=True),
                    )
                else: Pty_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")

                data_legend = [prp_scatter, prE_scatter, Eop_scatter, Acc_scatter, Eq1_scatter, Pty_scatter]

                fig = go.Figure(data=data_legend)

                fig.update_yaxes(title='Fairness', title_font=dict(size=20), visible=True)
                fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
                fig.update_traces(line_shape="vh")
                fig.update_layout(yaxis = dict(tickfont = dict(size=15)), xaxis = dict(tickfont = dict(size=15)))

                st.plotly_chart(fig)

            if checkLR and opt_method == "MOBO":

                n_features = len(X.columns)

                for notion in notions2:

                    class MLP:
                        @property
                        def configspace(self) -> ConfigurationSpace:
                            cs = ConfigurationSpace()

                            # First we create our hyperparameters
                            solver = Categorical("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], default="newton-cg")
                            C = Float("C", (0.001, 1000.0), default=1.0, log=True)
                            penalty = Categorical("penalty", ['l1', 'l2', 'elasticnet'], default='l2')
                            fit_intercept = Categorical("fit_intercept", [True, False], default=True)
                            l1_ratio = Float("l1_ratio", (0.0, 1.0), default=0.0)

                            # Then we create dependencies
                            use_l1_ratio = InCondition(child=l1_ratio, parent=penalty, values=["elasticnet"])
                            
                            penalty_and_lbfgs = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "l1"),
                                                ForbiddenEqualsClause(solver, "lbfgs")
                                                )
                            
                            penalty_and_lbfgs2 = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "elasticnet"),
                                                ForbiddenEqualsClause(solver, "lbfgs")
                                                )
                            
                            penalty_and_libinear = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "elasticnet"),
                                                ForbiddenEqualsClause(solver, "liblinear")
                                                )
                            
                            penalty_and_newtoncg = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "l1"),
                                                ForbiddenEqualsClause(solver, "newton-cg")
                                                )
                            
                            penalty_and_newtoncg2 = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "elasticnet"),
                                                ForbiddenEqualsClause(solver, "newton-cg")
                                                )
                            
                            penalty_and_newtonchol = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "l1"),
                                                ForbiddenEqualsClause(solver, "newton-cholesky")
                                                )
                            
                            penalty_and_newtonchol2 = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "elasticnet"),
                                                ForbiddenEqualsClause(solver, "newton-cholesky")
                                                )
                            
                            penalty_and_sag = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "elasticnet"),
                                                ForbiddenEqualsClause(solver, "sag")
                                                )
                            
                            penalty_and_sag2 = ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(penalty, "l1"),
                                                ForbiddenEqualsClause(solver, "sag")
                                                )
                            
                            # Add hyperparameters and conditions to our configspace
                            cs.add_hyperparameters([solver, C, penalty, fit_intercept, l1_ratio])
                            cs.add_conditions([use_l1_ratio])
                            cs.add_forbidden_clauses([penalty_and_lbfgs, penalty_and_lbfgs2,
                                                    penalty_and_libinear, 
                                                    penalty_and_newtoncg, penalty_and_newtoncg2,
                                                    penalty_and_newtonchol, penalty_and_newtonchol2,
                                                    penalty_and_sag, penalty_and_sag2])

                            return cs

                        def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:

                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                
                                classifier = LogisticRegression(
                                    solver=config["solver"],
                                    C=config["C"],
                                    penalty=config["penalty"],
                                    fit_intercept=config["fit_intercept"],
                                    l1_ratio=config["l1_ratio"],
                                    random_state=seed,
                                )
                                
                                test_scores = []
                                fair = []

                                for seed in range(floor(n_seeds/2)):
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
                                    X_privileged_test = X_test[(X_test['Privileged'] > 0)]
                                    X_not_privileged_test = X_test[(X_test['Privileged'] == 0)]
                                    y_privileged_test = y_test[(X_test['Privileged'] > 0)]
                                    y_not_privileged_test = y_test[(X_test['Privileged'] == 0)]
                                    classifier.fit(X_train, y_train)
                                    test_scores.append(classifier.score(X_test, y_test))
                                    fair.append(fairness(X_privileged_test, y_privileged_test, 
                                                    X_not_privileged_test, y_not_privileged_test, 
                                                    classifier, False))
                                    
                                fairness_metrics = np.array(fair)

                            return {
                                "accuracy": 1 - np.mean(test_scores),
                                notion: np.mean(fairness_metrics[:, notions.index(notion)]),
                            }

                    def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
                        """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
                        average_costs = []
                        average_pareto_costs = []
                        for config in smac.runhistory.get_configs():
                            # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
                            average_cost = smac.runhistory.average_cost(config)

                            if config in incumbents:
                                average_pareto_costs += [average_cost]
                            else:
                                average_costs += [average_cost]

                        # Let's work with a numpy array
                        costs = np.vstack(average_costs)
                        pareto_costs = np.vstack(average_pareto_costs)
                        pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

                        costs_x, costs_y = costs[:, 0], costs[:, 1]
                        pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]
                        
                        plot_x = 1-costs_x
                        plot_y = 1-costs_y
                        plot_x_pareto = 1-pareto_costs_x
                        plot_y_pareto = 1-pareto_costs_y

                        plt.scatter(plot_x, plot_y, marker="x", label="Configuration")
                        plt.scatter(plot_x_pareto, plot_y_pareto, marker="x", c="r", label="Incumbent")
                        plt.step(
                            [plot_x_pareto[0]] + plot_x_pareto.tolist() + [np.min(plot_x)],  # We add bounds
                            [np.min(plot_y)] + plot_y_pareto.tolist() + [np.max(plot_y_pareto)],  # We add bounds
                            where="post",
                            linestyle=":",
                        )

                        plt.title("Pareto-Front")
                        plt.xlabel(smac.scenario.objectives[0])
                        plt.ylabel(smac.scenario.objectives[1])
                        plt.legend()
                        plt.show()


                    if __name__ == "__main__":
                        mlp = MLP()
                        objectives = ["accuracy", notion]

                        # Define our environment variables
                        scenario = Scenario(
                            mlp.configspace,
                            objectives=objectives,
                            walltime_limit=60000,  # After 30 seconds, we stop the hyperparameter optimization
                            n_trials=100,  # Evaluate max 200 different trials
                            n_workers=1,
                        )
                        
                        initial_design = LatinHypercubeInitialDesign(scenario, n_configs_per_hyperparameter=6)

                        # We want to run five random configurations before starting the optimization.
                        #initial_design = HPOFacade.get_initial_design(scenario, n_configs_per_hyperparameter=6)
                        multi_objective_algorithm = ParEGO(scenario)
                        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

                        # Create our SMAC object and pass the scenario and the train method
                        smac = HPOFacade(
                            scenario,
                            mlp.train,
                            initial_design=initial_design,
                            multi_objective_algorithm=multi_objective_algorithm,
                            intensifier=intensifier,
                            overwrite=True,
                        )

                        # Let's optimize
                        incumbents = smac.optimize()

                        # Get cost of default configuration
                        default_cost = smac.validate(mlp.configspace.get_default_configuration())
                        print(f"Validated costs from default config: \n--- {default_cost}\n")

                        print("Validated costs from the Pareto front (incumbents):")
                        for incumbent in incumbents:
                            cost = smac.validate(incumbent)
                            print("---", cost)

                        # Let's plot a pareto front
                        plot_pareto(smac, incumbents)

                    solver = []
                    C = []
                    penalty = []
                    fit_intercept = []
                    solver = []
                    l1_ratio = []
                    accuracy = []
                    prp = []

                    for i in range(len(incumbents)):
                        solver.append(incumbents[i]['solver'])
                        C.append(incumbents[i]['C'])
                        penalty.append(incumbents[i]['penalty'])
                        fit_intercept.append(incumbents[i]['fit_intercept'])
                        l1_ratio.append(incumbents[i]['l1_ratio'])
                        accuracy.append(1-smac.validate(incumbents[i])[0])
                        prp.append(1-smac.validate(incumbents[i])[1])
                        
                    solutions_space = {
                        'accuracy': accuracy, 
                        notion: prp,
                        'solver': solver,
                        'C': C,
                        'penalty': penalty,
                        'fit_intercept': fit_intercept,
                        'l1_ratio': l1_ratio,
                        }

                    dfp_lr = pd.DataFrame(data=solutions_space)
                    dfp_lr.sort_values('accuracy', inplace=True, ignore_index=True)
                    dfp_lr.replace(to_replace=[None], value='None', inplace=True)

                    if notion=='predictive parity':
                        df_MODEL_prp[2] = dfp_lr
                    if notion=='predictive equality':
                        df_MODEL_prE[2] = dfp_lr
                    if notion=='equal opportunity':
                        df_MODEL_Eop[2] = dfp_lr
                    if notion=='accuracy equality':
                        df_MODEL_Acc[2] = dfp_lr
                    if notion=='equalized odds':
                        df_MODEL_Eq1[2] = dfp_lr
                    if notion=='statistical parity':
                        df_MODEL_Pty[2] = dfp_lr

                    fig = px.line(dfp_lr, x = 'accuracy', y = notion,
                                    hover_data=['solver','C','penalty','fit_intercept','l1_ratio'])
                    fig.update_traces(mode="markers+lines", line_shape="vh", line_dash="dash")
                    fig.update_traces(marker=dict(line=dict(width=1.2, color='DarkSlateGrey')))

                    fig.update_yaxes(title_font=dict(size=20))
                    fig.update_xaxes(title_font=dict(size=20))

                    st.subheader('Logistic Regression:' + notion)

                    st.plotly_chart(fig)

                    st.write(dfp_lr.head())

                    csv = convert_df(dfp_lr)

                    download_button_str = download_button(csv, "lr_MOBO.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)

                    st.write('---')

            #######################################
            #######################################
            ###--------- RANDOM FOREST ---------###
            #######################################
            #######################################

            if checkRF and opt_method=='Grid Search': 

                test_scores = []
                test_scores_privileged = []
                test_scores_not_privileged = []

                mean_test_scores = []
                mean_test_scores_privileged = []
                mean_test_scores_not_privileged = []

                sem_test_scores = []

                fairness_metrics = []

                mean_prp = []
                mean_prE = []
                mean_Eop = []
                mean_Acc = []
                mean_Eq1 = []
                mean_Pty = []

                sem_prp = []
                sem_prE = []
                sem_Eop = []
                sem_Acc = []
                sem_Eq1 = []
                sem_Pty = []

                hyp_min_samples_leaf = []
                hyp_min_samples_split = []
                hyp_max_depth = []
                hyp_max_features = []
                hyp_criterions = []
                hyp_class_weights = []
                hyp_bootstrap = []
                hyp_max_samples = []

                output = []
                i = 0
                total_combinations = (len(bootstrap) * len(max_samples) * len(RF_class_weight) * len(RF_criterion) * len(RF_max_features) * RF_min_samples_split.size * RF_min_samples_leaf.size * RF_max_depth.size)*(len(max_samples))/(len(max_samples)+1)

                progress_text = "Fitting random forests..."
                my_bar = st.progress(0, text=progress_text)

                cond_max_samples = [None]

                for w in RF_class_weight :
                    for depth in RF_max_depth :
                        for b in bootstrap:
                            if b==False: max_samples2 = cond_max_samples
                            else: max_samples2 = max_samples
                            for max_sample in max_samples2 :
                                for c in RF_criterion :
                                    for features in RF_max_features :
                                        for samples_split in RF_min_samples_split :
                                            for samples_leaf in RF_min_samples_leaf :
                                                for seed in range(n_seeds) :
                                                    
                                                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=seed, train_size=split_size)
                                                    X_privileged_test = X_test[(X_test[X_sen] > 0)]
                                                    X_not_privileged_test = X_test[(X_test[X_sen] == 0)]
                                                    y_privileged_test = y_test[(X_test[X_sen] > 0)]
                                                    y_not_privileged_test = y_test[(X_test[X_sen] == 0)]
                                                    rfc = RandomForestClassifier(class_weight=w, bootstrap=b, max_depth=depth, max_samples=max_sample, criterion=c, max_features=features, min_samples_split=samples_split, min_samples_leaf=samples_leaf, random_state=0)
                                                    rfc.fit(X_train, y_train)
                                                    test_scores.append(rfc.score(X_test, y_test))
                                                    test_scores_privileged.append(rfc.score(X_privileged_test, y_privileged_test))
                                                    test_scores_not_privileged.append(rfc.score(X_not_privileged_test, y_not_privileged_test))

                                                    output.append(fc.fairness(X_privileged_test, y_privileged_test, X_not_privileged_test, y_not_privileged_test, rfc, False))

                                                fairness_metrics = np.array(output)

                                                ## Fairness metrics
                                                mean_prp.append(fairness_metrics[:, 0].mean())
                                                mean_prE.append(fairness_metrics[:, 1].mean())
                                                mean_Eop.append(fairness_metrics[:, 2].mean())
                                                mean_Acc.append(fairness_metrics[:, 3].mean())
                                                mean_Eq1.append(fairness_metrics[:, 4].mean())
                                                mean_Pty.append(fairness_metrics[:, 5].mean())

                                                sem_prp.append(sem(fairness_metrics[:, 0]))
                                                sem_prE.append(sem(fairness_metrics[:, 1]))
                                                sem_Eop.append(sem(fairness_metrics[:, 2]))
                                                sem_Acc.append(sem(fairness_metrics[:, 3]))
                                                sem_Eq1.append(sem(fairness_metrics[:, 4]))
                                                sem_Pty.append(sem(fairness_metrics[:, 5]))

                                                ## Accuracy scores
                                                mean_test_scores.append(mean(test_scores))
                                                mean_test_scores_privileged.append(mean(test_scores_privileged))
                                                mean_test_scores_not_privileged.append(mean(test_scores_not_privileged))

                                                sem_test_scores.append(sem(test_scores))

                                                ## Hyperparameters
                                                hyp_min_samples_leaf.append(samples_leaf)
                                                hyp_min_samples_split.append(samples_split)
                                                hyp_max_depth.append(depth)
                                                hyp_max_features.append(features)
                                                hyp_criterions.append(c)
                                                hyp_class_weights.append(w)
                                                hyp_bootstrap.append(b)
                                                hyp_max_samples.append(max_sample)

                                                ## Reset for each hyperparameter combination
                                                output = []
                                                test_scores = []
                                                test_scores_privileged = []
                                                test_scores_not_privileged = []
                                                i = i + 1
                                                my_bar.progress(i/total_combinations, text=progress_text)

                my_bar.progress(0.999, text=progress_text)
                st.subheader('Random Forest Classifier: Results')

                solutions_space = {
                    'min_samples_leaf': hyp_min_samples_leaf, 
                    'min_samples_split': hyp_min_samples_split,
                    'max_depth': hyp_max_depth,
                    'max_features': hyp_max_features,
                    'criterion': hyp_criterions,
                    'class_weight': hyp_class_weights,
                    'bootstrap': hyp_bootstrap,
                    'max_samples': hyp_max_samples,
                    'Accuracy': mean_test_scores,
                    'Accuracy std': sem_test_scores,
                    'Predictive parity': mean_prp,
                    'Predictive parity SEM': sem_prp,
                    'Predictive equality': mean_prE,
                    'Predictive equality SEM': sem_prE,
                    'Equal opportunity': mean_Eop,
                    'Equal opportunity SEM': sem_Eop,
                    'Accuracy equality': mean_Acc,
                    'Accuracy equality SEM': sem_Acc,
                    'Equalized odds': mean_Eq1,
                    'Equalized odds SEM': sem_Eq1,
                    'Statistical parity': mean_Pty,
                    'Statistical parity SEM': sem_Pty,
                }

                df_rf = pd.DataFrame(data=solutions_space)
                df_rf.replace(to_replace=[None], value='None', inplace=True)
                df_MODEL[3] = df_rf
                st.write(df_rf.head(10))

                csv = convert_df(df_rf)

                download_button_str = download_button(csv, "rf_full_table.csv", 'Download Table', pickle_it=False)
                st.markdown(download_button_str, unsafe_allow_html=True)

                st.write('---')

                if prp_ok:
                    df_rf_prp = pt.pareto_frontier(Xs=df_rf['Accuracy'], Ys=df_rf['Predictive parity'], name='Predictive parity', maxX=True, maxY=True)
                if prE_ok:
                    df_rf_prE = pt.pareto_frontier(Xs=df_rf['Accuracy'], Ys=df_rf['Predictive equality'], name='Predictive equality', maxX=True, maxY=True)
                if Eop_ok:
                    df_rf_Eop = pt.pareto_frontier(Xs=df_rf['Accuracy'], Ys=df_rf['Equal opportunity'], name='Equal opportunity', maxX=True, maxY=True)
                if Acc_ok:    
                    df_rf_Acc = pt.pareto_frontier(Xs=df_rf['Accuracy'], Ys=df_rf['Accuracy equality'], name='Accuracy equality', maxX=True, maxY=True)
                if Eq1_ok:
                    df_rf_Eq1 = pt.pareto_frontier(Xs=df_rf['Accuracy'], Ys=df_rf['Equalized odds'], name='Equalized odds', maxX=True, maxY=True)
                if Pty_ok:
                    df_rf_Pty = pt.pareto_frontier(Xs=df_rf['Accuracy'], Ys=df_rf['Statistical parity'], name='Statistical parity', maxX=True, maxY=True)

                mean_test_scores_np = np.array(mean_test_scores)
                sem_test_scores_np = np.array(sem_test_scores)
                prp_np = np.array(mean_prp)
                prE_np = np.array(mean_prE)
                Eop_np = np.array(mean_Eop)
                Acc_np = np.array(mean_Acc)
                Eq1_np = np.array(mean_Eq1)
                Pty_np = np.array(mean_Pty)

                sem_prp_np = np.array(sem_prp)
                sem_prE_np = np.array(sem_prE)
                sem_Eop_np = np.array(sem_Eop)
                sem_Acc_np = np.array(sem_Acc)
                sem_Eq1_np = np.array(sem_Eq1)
                sem_Pty_np = np.array(sem_Pty)

                if prp_ok:
                    st.subheader('RF - Predictive Parity')
                    df_rf_prp = SingleModelParetoRF(df_rf_prp, prp_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_bootstrap, hyp_max_depth, hyp_max_samples, df_rf, 'Predictive parity')
                    df_rf_prp.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prp[3] = df_rf_prp
                    st.write(df_rf_prp)
                    csv = convert_df(df_rf_prp)

                    download_button_str = download_button(csv, "rf_pareto_prp.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if prE_ok:
                    st.subheader('RF - Predictive Equality')
                    df_rf_prE = SingleModelParetoRF(df_rf_prE, prE_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_bootstrap, hyp_max_depth, hyp_max_samples, df_rf, 'Predictive equality')
                    df_rf_prE.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_prE[3] = df_rf_prE
                    st.write(df_rf_prE)
                    csv = convert_df(df_rf_prE)
                    download_button_str = download_button(csv, "rf_pareto_prE.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eop_ok:
                    st.subheader('RF - Equal Opportunity')
                    df_rf_Eop = SingleModelParetoRF(df_rf_Eop, Eop_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_bootstrap, hyp_max_depth, hyp_max_samples, df_rf, 'Equal opportunity')
                    df_rf_Eop.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eop[3] = df_rf_Eop
                    st.write(df_rf_Eop)
                    csv = convert_df(df_rf_Eop)
                    download_button_str = download_button(csv, "rf_pareto_Eop.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Acc_ok:
                    st.subheader('RF - Accuracy Equality')
                    df_rf_Acc = SingleModelParetoRF(df_rf_Acc, Acc_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_bootstrap, hyp_max_depth, hyp_max_samples, df_rf, 'Accuracy equality')
                    df_rf_Acc.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Acc[3] = df_rf_Acc
                    st.write(df_rf_Acc)
                    csv = convert_df(df_rf_Acc)
                    download_button_str = download_button(csv, "rf_pareto_Acc.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Eq1_ok:
                    st.subheader('RF - Equalized Odds')
                    df_rf_Eq1 = SingleModelParetoRF(df_rf_Eq1, Eq1_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_bootstrap, hyp_max_depth, hyp_max_samples, df_rf, 'Equalized odds')
                    df_rf_Eq1.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Eq1[3] = df_rf_Eq1
                    st.write(df_rf_Eq1)
                    csv = convert_df(df_rf_Eq1)
                    download_button_str = download_button(csv, "rf_pareto_Eq1.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                if Pty_ok:
                    st.subheader('DT - Statistical Parity')
                    df_rf_Pty = SingleModelParetoRF(df_rf_Pty, Pty_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                mean_test_scores_np, sem_test_scores_np,
                                                hyp_min_samples_leaf, hyp_min_samples_split, hyp_max_features, hyp_criterions, hyp_class_weights, hyp_bootstrap, hyp_max_depth, hyp_max_samples, df_rf, 'Statistical parity')
                    df_rf_Pty.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL_Pty[3] = df_rf_Pty
                    st.write(df_rf_Pty)
                    csv = convert_df(df_rf_Pty)
                    download_button_str = download_button(csv, "rf_pareto_Pty.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                st.subheader('RF - Fairness Metrics Comparison')

                if prp_ok:
                    prp_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_rf_prp['Accuracy'],
                            y=df_rf_prp['Predictive parity'],
                            name='Predictive parity',
                            marker=dict(size=6, color="red", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_rf_prp['min_samples_leaf'], df_rf_prp['min_samples_split'], df_rf_prp['max_features'], 
                                                    df_rf_prp['criterion'], df_rf_prp['class weight'], df_rf_prp['bootstrap'], df_rf_prp['max_depth'], df_rf_prp['max_samples']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Predictive parity: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            'bootstrap: %{customdata[5]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_rf_prp['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_rf_prp['Predictive parity SEM'], visible=True),
                            )
                else: prp_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if prE_ok:
                    prE_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_rf_prE['Accuracy'],
                            y=df_rf_prE['Predictive equality'],
                            name='Predictive equality',
                            marker=dict(size=6, color="green", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_rf_prE['min_samples_leaf'], df_rf_prE['min_samples_split'], df_rf_prE['max_features'], 
                                                    df_rf_prE['criterion'], df_rf_prE['class weight'], df_rf_prE['bootstrap'], df_rf_prE['max_depth'], df_rf_prE['max_samples']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Predictive equality: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            'bootstrap: %{customdata[5]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_rf_prE['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_rf_prE['Predictive equality SEM'], visible=True),
                        )
                else: prE_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Eop_ok:
                    Eop_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_rf_Eop['Accuracy'],
                            y=df_rf_Eop['Equal opportunity'],
                            name='Equal opportunity',
                            marker=dict(size=6, color="gold", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_rf_Eop['min_samples_leaf'], df_rf_Eop['min_samples_split'], df_rf_Eop['max_features'], 
                                                    df_rf_Eop['criterion'], df_rf_Eop['class weight'], df_rf_Eop['bootstrap'], df_rf_Eop['max_depth'], df_rf_Eop['max_samples']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Equal opportunity: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            'bootstrap: %{customdata[5]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_rf_Eop['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_rf_Eop['Equal opportunity SEM'], visible=True),
                        )
                else: Eop_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                    
                if Acc_ok:
                    Acc_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_rf_Acc['Accuracy'],
                            y=df_rf_Acc['Accuracy equality'],
                            name='Accuracy equality',
                            marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_rf_Acc['min_samples_leaf'], df_rf_Acc['min_samples_split'], df_rf_Acc['max_features'], 
                                                    df_rf_Acc['criterion'], df_rf_Acc['class weight'], df_rf_Acc['bootstrap'], df_rf_Acc['max_depth'], df_rf_Acc['max_samples']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Accuracy equality: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            'bootstrap: %{customdata[5]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_rf_Acc['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_rf_Acc['Accuracy equality SEM'], visible=True),
                        )
                else: Acc_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Eq1_ok:
                    Eq1_scatter = go.Scatter(
                            mode='lines+markers',
                            x=df_rf_Eq1['Accuracy'],
                            y=df_rf_Eq1['Equalized odds'],
                            name='Equalized odds',
                            marker=dict(size=6, color="black", symbol='circle',line=dict(width=2,
                                                        color='darkslategray')),
                            line=dict(width=2),
                            customdata = np.stack((df_rf_Eq1['min_samples_leaf'], df_rf_Eq1['min_samples_split'], df_rf_Eq1['max_features'], 
                                                    df_rf_Eq1['criterion'], df_rf_Eq1['class weight'], df_rf_Eq1['bootstrap'], df_rf_Eq1['max_depth'], df_rf_Eq1['max_samples']), axis=-1),
                            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                            'Equalized odds: %{y:,.4f} <br>' + 
                            'min_samples_leaf: %{customdata[0]} <br>' +
                            'min_samples_split: %{customdata[1]} <br>' +
                            'max_features: %{customdata[2]} <br>' +
                            'criterion: %{customdata[3]} <br>' +
                            'class_weight: %{customdata[4]} <br>' +
                            'bootstrap: %{customdata[5]} <br>' +
                            '<extra></extra>'),
                            error_x = dict(type='data', array=df_rf_Eq1['Accuracy SEM'], visible=True),
                            error_y = dict(type='data', array=df_rf_Eq1['Equalized odds SEM'], visible=True),
                            )
                else: Eq1_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                if Pty_ok:
                    Pty_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_rf_Pty['Accuracy'],
                                y=df_rf_Pty['Statistical parity'],
                                name='Statistical parity',
                                marker=dict(size=6, color="blue", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_rf_Pty['min_samples_leaf'], df_rf_Pty['min_samples_split'], df_rf_Pty['max_features'], 
                                                    df_rf_Pty['criterion'], df_rf_Pty['class weight'], df_rf_Pty['bootstrap'], df_rf_Pty['max_depth'], df_rf_Pty['max_samples']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'min_samples_leaf: %{customdata[0]} <br>' +
                                'min_samples_split: %{customdata[1]} <br>' +
                                'max_features: %{customdata[2]} <br>' +
                                'criterion: %{customdata[3]} <br>' +
                                'class_weight: %{customdata[4]} <br>' +
                                'bootstrap: %{customdata[5]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_rf_Pty['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_rf_Pty['Predictive parity SEM'], visible=True),
                    )
                else: Pty_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")

                data_legend = [prp_scatter, prE_scatter, Eop_scatter, Acc_scatter, Eq1_scatter, Pty_scatter]

                fig = go.Figure(data=data_legend)

                fig.update_yaxes(title='Fairness', title_font=dict(size=20), visible=True)
                fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
                fig.update_traces(line_shape="vh")
                fig.update_layout(yaxis = dict(tickfont = dict(size=14)), xaxis = dict(tickfont = dict(size=15)))

                st.plotly_chart(fig)

            

            if checkRF and opt_method=='MOBO': 

                n_features = len(X.columns)

                for notion in notions2:

                    class MLP:
                        @property
                        def configspace(self) -> ConfigurationSpace:
                            cs = ConfigurationSpace()

                            criterion = Categorical("criterion", ["gini", "entropy", "log_loss"], default="gini")
                            max_depth = Integer("max_depth", (1, 100), default=100)
                            max_features = Integer("max_features", (1, n_features), default=n_features)
                            min_samples_split = Integer("min_samples_split", (2, 40), default=2)
                            min_samples_leafs = Integer("min_samples_leaf", (1, 40), default=1)
                            #class_weight = Categorical("class_weights", ["balanced", "balanced_subsample"], default="balanced")
                            bootstrap = Categorical("bootstrap", [True, False], default=True)
                            max_samples = Float("max_samples", (0.5, 1.0), default=1.0)
                            #ccp_alpha = Float("ccp_alpha", (0.0, 1.0), default=0.0)
                            
                            use_max_samples = InCondition(child=max_samples, parent=bootstrap, values=[True])

                            cs.add_hyperparameters([criterion, max_depth, max_features, min_samples_split, min_samples_leafs, bootstrap, max_samples])
                            cs.add_conditions([use_max_samples])
                            return cs

                        def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:

                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                
                                classifier = RandomForestClassifier(
                                criterion=config["criterion"],
                                max_features=config["max_features"],
                                min_samples_split=config["min_samples_split"],
                                min_samples_leaf=config["min_samples_leaf"],
                                max_depth=config["max_depth"],
                                bootstrap=config["bootstrap"],
                                max_samples=config["max_samples"],
                                random_state=0,
                                )
                                
                                test_scores = []
                                fair = []

                                for seed in range(floor(n_seeds/2)):
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
                                    X_privileged_test = X_test[(X_test['Privileged'] > 0)]
                                    X_not_privileged_test = X_test[(X_test['Privileged'] == 0)]
                                    y_privileged_test = y_test[(X_test['Privileged'] > 0)]
                                    y_not_privileged_test = y_test[(X_test['Privileged'] == 0)]
                                    classifier.fit(X_train, y_train)
                                    test_scores.append(classifier.score(X_test, y_test))
                                    fair.append(fairness(X_privileged_test, y_privileged_test, 
                                                    X_not_privileged_test, y_not_privileged_test, 
                                                    classifier, False))
                                    
                                fairness_metrics = np.array(fair)

                            return {
                                "accuracy": 1 - np.mean(test_scores),
                                notion: np.mean(fairness_metrics[:, notions.index(notion)]),
                            }

                    def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
                        """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
                        average_costs = []
                        average_pareto_costs = []
                        for config in smac.runhistory.get_configs():
                            # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
                            average_cost = smac.runhistory.average_cost(config)

                            if config in incumbents:
                                average_pareto_costs += [average_cost]
                            else:
                                average_costs += [average_cost]

                        # Let's work with a numpy array
                        costs = np.vstack(average_costs)
                        pareto_costs = np.vstack(average_pareto_costs)
                        pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

                        costs_x, costs_y = costs[:, 0], costs[:, 1]
                        pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]
                        
                        plot_x = 1-costs_x
                        plot_y = 1-costs_y
                        plot_x_pareto = 1-pareto_costs_x
                        plot_y_pareto = 1-pareto_costs_y

                        plt.scatter(plot_x, plot_y, marker="x", label="Configuration")
                        plt.scatter(plot_x_pareto, plot_y_pareto, marker="x", c="r", label="Incumbent")
                        plt.step(
                            [plot_x_pareto[0]] + plot_x_pareto.tolist() + [np.min(plot_x)],  # We add bounds
                            [np.min(plot_y)] + plot_y_pareto.tolist() + [np.max(plot_y_pareto)],  # We add bounds
                            where="post",
                            linestyle=":",
                        )

                        plt.title("Pareto-Front")
                        plt.xlabel(smac.scenario.objectives[0])
                        plt.ylabel(smac.scenario.objectives[1])
                        plt.legend()
                        plt.show()


                    if __name__ == "__main__":
                        mlp = MLP()
                        objectives = ["accuracy", notion]

                        # Define our environment variables
                        scenario = Scenario(
                            mlp.configspace,
                            objectives=objectives,
                            walltime_limit=60000,  # After 30 seconds, we stop the hyperparameter optimization
                            n_trials=100,  # Evaluate max 200 different trials
                            n_workers=1,
                        )
                        
                        initial_design = LatinHypercubeInitialDesign(scenario, n_configs_per_hyperparameter=6)

                        # We want to run five random configurations before starting the optimization.
                        #initial_design = HPOFacade.get_initial_design(scenario, n_configs_per_hyperparameter=6)
                        multi_objective_algorithm = ParEGO(scenario)
                        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

                        # Create our SMAC object and pass the scenario and the train method
                        smac = HPOFacade(
                            scenario,
                            mlp.train,
                            initial_design=initial_design,
                            multi_objective_algorithm=multi_objective_algorithm,
                            intensifier=intensifier,
                            overwrite=True,
                        )

                        # Let's optimize
                        incumbents = smac.optimize()

                        # Get cost of default configuration
                        default_cost = smac.validate(mlp.configspace.get_default_configuration())
                        print(f"Validated costs from default config: \n--- {default_cost}\n")

                        print("Validated costs from the Pareto front (incumbents):")
                        for incumbent in incumbents:
                            cost = smac.validate(incumbent)
                            print("---", cost)

                        # Let's plot a pareto front
                        plot_pareto(smac, incumbents)

                    criterion = []
                    max_depth = []
                    max_features = []
                    min_samples_leaf = []
                    min_samples_split = []
                    max_samples = []
                    bootstrap = []
                    accuracy = []
                    prp = []

                    for i in range(len(incumbents)):
                        
                        criterion.append(incumbents[i]['criterion'])
                        max_depth.append(incumbents[i]['max_depth'])
                        max_features.append(incumbents[i]['max_features'])
                        min_samples_leaf.append(incumbents[i]['min_samples_leaf'])
                        min_samples_split.append(incumbents[i]['min_samples_split'])
                        max_samples.append(incumbents[i]['max_samples'])
                        bootstrap.append(incumbents[i]['bootstrap'])
                        accuracy.append(1-smac.validate(incumbents[i])[0])
                        prp.append(1-smac.validate(incumbents[i])[1])
                        
                    solutions_space = {
                        'accuracy': accuracy, 
                        notion: prp,
                        'criterion': criterion,
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': min_samples_split,
                        'max_samples': max_samples,
                        'bootstrap': bootstrap,
                        #'ccp_alpha': ccp_alpha,
                        }

                    dfp_rfc = pd.DataFrame(data=solutions_space)
                    dfp_rfc.sort_values('accuracy', inplace=True, ignore_index=True)
                    dfp_rfc.replace(to_replace=[None], value='None', inplace=True)
                    
                    if notion=='predictive parity':
                        df_MODEL_prp[3] = dfp_rfc
                    if notion=='predictive equality':
                        df_MODEL_prE[3] = dfp_rfc
                    if notion=='equal opportunity':
                        df_MODEL_Eop[3] = dfp_rfc
                    if notion=='accuracy equality':
                        df_MODEL_Acc[3] = dfp_rfc
                    if notion=='equalized odds':
                        df_MODEL_Eq1[3] = dfp_rfc
                    if notion=='statistical parity':
                        df_MODEL_Pty[3] = dfp_rfc

                    st.subheader('Random Forest Classifier:' + notion)

                    fig = px.line(dfp_rfc, x = 'accuracy', y = notion,
                                    hover_data=['min_samples_leaf','min_samples_split','criterion',
                                                'max_features','max_depth', 'bootstrap', 'max_samples'])
                    fig.update_traces(mode="markers+lines", line_shape="vh", line_dash="dash")
                    fig.update_traces(marker=dict(line=dict(width=1.2, color='DarkSlateGrey')))

                    fig.update_yaxes(title_font=dict(size=20))
                    fig.update_xaxes(title_font=dict(size=20))

                    st.plotly_chart(fig)

                    st.write(dfp_rfc.head())

                    csv = convert_df(dfp_rfc)

                    download_button_str = download_button(csv, "dtc_MOBO.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)

                    st.write('---')    
                    
                
            ########################################
            ########################################
            ###--------- NEURAL NETWORK ---------###
            ########################################
            ########################################

            if opt_method == 'Grid Search':
                if checkNN :

                    test_scores = []
                    test_scores_privileged = []
                    test_scores_not_privileged = []

                    mean_test_scores = []
                    mean_test_scores_privileged = []
                    mean_test_scores_not_privileged = []

                    sem_test_scores = []

                    fairness_metrics = []

                    mean_prp = []
                    mean_prE = []
                    mean_Eop = []
                    mean_Acc = []
                    mean_Eq1 = []
                    mean_Pty = []

                    sem_prp = []
                    sem_prE = []
                    sem_Eop = []
                    sem_Acc = []
                    sem_Eq1 = []
                    sem_Pty = []

                    hyp_L1_nodes = []
                    hyp_L2_nodes = []
                    hyp_L1_dropout_rate = []
                    hyp_L2_dropout_rate = []
                    hyp_batch_size = []
                    hyp_epochs = []

                    output = []
                    i = 0
                    total_combinations = L1_nodes.size * L2_nodes.size * L1_dropout_rates.size * L2_dropout_rates.size * batch_sizes.size * epochs.size

                    progress_text = "Fitting neural networks..."
                    my_bar = st.progress(0, text=progress_text)

                    for L1_node in L1_nodes :
                        for L2_node in L2_nodes:
                            for L1_dropout_rate in L1_dropout_rates :
                                for L2_dropout_rate in L2_dropout_rates :
                                    for batch_size in batch_sizes :
                                        for epoch in epochs :
                                            for seed in range(n_seeds) :
                                                
                                                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=seed, train_size=split_size)
                                                X_privileged_test = X_test[(X_test[X_sen] > 0)]
                                                X_not_privileged_test = X_test[(X_test[X_sen] == 0)]
                                                y_privileged_test = y_test[(X_test[X_sen] > 0)]
                                                y_not_privileged_test = y_test[(X_test[X_sen] == 0)]
                                                
                                                NN = keras.Sequential([
                                                keras.layers.Dense(L1_node, input_shape=(X_scaled.shape[1],), activation='relu'),
                                                keras.layers.Dropout(L1_dropout_rate),
                                                keras.layers.Dense(L2_node, activation='relu'),
                                                keras.layers.Dropout(L2_dropout_rate),
                                                keras.layers.Dense(2, activation='relu')])

                                                NN.compile(optimizer='adam',
                                                            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                                            metrics=['accuracy'])

                                                NN.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=0)
                                                test_scores.append(NN.evaluate(X_test, y_test, verbose=0))
                                                test_scores_privileged.append(NN.evaluate(X_privileged_test, y_privileged_test, verbose=0))
                                                test_scores_not_privileged.append(NN.evaluate(X_not_privileged_test, y_not_privileged_test, verbose=0))
                                                output.append(fc.fairness(X_privileged_test, y_privileged_test, X_not_privileged_test, y_not_privileged_test, NN, True))

                                            fairness_metrics = np.array(output)

                                            ## Fairness metrics
                                            mean_prp.append(fairness_metrics[:, 0].mean())
                                            mean_prE.append(fairness_metrics[:, 1].mean())
                                            mean_Eop.append(fairness_metrics[:, 2].mean())
                                            mean_Acc.append(fairness_metrics[:, 3].mean())
                                            mean_Eq1.append(fairness_metrics[:, 4].mean())
                                            mean_Pty.append(fairness_metrics[:, 5].mean())

                                            sem_prp.append(sem(fairness_metrics[:, 0]))
                                            sem_prE.append(sem(fairness_metrics[:, 1]))
                                            sem_Eop.append(sem(fairness_metrics[:, 2]))
                                            sem_Acc.append(sem(fairness_metrics[:, 3]))
                                            sem_Eq1.append(sem(fairness_metrics[:, 4]))
                                            sem_Pty.append(sem(fairness_metrics[:, 5]))

                                            ## Accuracy scores
                                            mean_test_scores.append(mean(np.array(test_scores)[:,1]))
                                            mean_test_scores_privileged.append(mean(np.array(test_scores_privileged)[:,1]))
                                            mean_test_scores_not_privileged.append(mean(np.array(test_scores_not_privileged)[:,1]))

                                            sem_test_scores.append(sem(np.array(test_scores)[:,1]))

                                            ## Hyperparameters
                                            hyp_L1_nodes.append(L1_node)
                                            hyp_L2_nodes.append(L2_node)
                                            hyp_L1_dropout_rate.append(L1_dropout_rate)
                                            hyp_L2_dropout_rate.append(L2_dropout_rate)
                                            hyp_batch_size.append(batch_size)
                                            hyp_epochs.append(epoch)

                                            ## Reset for each hyperparameter combination
                                            output = []
                                            test_scores = []
                                            test_scores_privileged = []
                                            test_scores_not_privileged = []
                                            i = i + 1
                                            my_bar.progress(i/total_combinations, text=progress_text)
                                    
                    st.subheader('Neural Network: Results')

                    solutions_space = {
                        'L1_nodes': hyp_L1_nodes, 
                        'L2_nodes': hyp_L2_nodes,
                        'L1_dropout_rate': hyp_L1_dropout_rate,
                        'L2_dropout_rate': hyp_L2_dropout_rate,
                        'batch_size': hyp_batch_size,
                        'epochs': hyp_epochs,
                        'Accuracy': mean_test_scores,
                        'Accuracy std': sem_test_scores,
                        'Predictive parity': mean_prp,
                        'Predictive parity SEM': sem_prp,
                        'Predictive equality': mean_prE,
                        'Predictive equality SEM': sem_prE,
                        'Equal opportunity': mean_Eop,
                        'Equal opportunity SEM': sem_Eop,
                        'Accuracy equality': mean_Acc,
                        'Accuracy equality SEM': sem_Acc,
                        'Equalized odds': mean_Eq1,
                        'Equalized odds SEM': sem_Eq1,
                        'Statistical parity': mean_Pty,
                        'Statistical parity SEM': sem_Pty,
                    }

                    df_NN = pd.DataFrame(data=solutions_space)
                    df_NN.replace(to_replace=[None], value='None', inplace=True)
                    df_MODEL[4] = df_NN
                    st.write(df_NN.head(10))

                    csv = convert_df(df_NN)

                    download_button_str = download_button(csv, "NN_full_table.csv", 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)

                    st.write('---')

                    if prp_ok:
                        df_NN_prp = pt.pareto_frontier(Xs=df_NN['Accuracy'], Ys=df_NN['Predictive parity'], name='Predictive parity', maxX=True, maxY=True)
                    if prE_ok:
                        df_NN_prE = pt.pareto_frontier(Xs=df_NN['Accuracy'], Ys=df_NN['Predictive equality'], name='Predictive equality', maxX=True, maxY=True)
                    if Eop_ok:
                        df_NN_Eop = pt.pareto_frontier(Xs=df_NN['Accuracy'], Ys=df_NN['Equal opportunity'], name='Equal opportunity', maxX=True, maxY=True)
                    if Acc_ok:    
                        df_NN_Acc = pt.pareto_frontier(Xs=df_NN['Accuracy'], Ys=df_NN['Accuracy equality'], name='Accuracy equality', maxX=True, maxY=True)
                    if Eq1_ok:
                        df_NN_Eq1 = pt.pareto_frontier(Xs=df_NN['Accuracy'], Ys=df_NN['Equalized odds'], name='Equalized odds', maxX=True, maxY=True)
                    if Pty_ok:
                        df_NN_Pty = pt.pareto_frontier(Xs=df_NN['Accuracy'], Ys=df_NN['Statistical parity'], name='Statistical parity', maxX=True, maxY=True)

                    mean_test_scores_np = np.array(mean_test_scores)
                    sem_test_scores_np = np.array(sem_test_scores)
                    prp_np = np.array(mean_prp)
                    prE_np = np.array(mean_prE)
                    Eop_np = np.array(mean_Eop)
                    Acc_np = np.array(mean_Acc)
                    Eq1_np = np.array(mean_Eq1)
                    Pty_np = np.array(mean_Pty)

                    sem_prp_np = np.array(sem_prp)
                    sem_prE_np = np.array(sem_prE)
                    sem_Eop_np = np.array(sem_Eop)
                    sem_Acc_np = np.array(sem_Acc)
                    sem_Eq1_np = np.array(sem_Eq1)
                    sem_Pty_np = np.array(sem_Pty)

                    if prp_ok:
                        st.subheader('NN - Predictive Parity')
                        df_NN_prp = SingleModelParetoNN(df_NN_prp, prp_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                    mean_test_scores_np, sem_test_scores_np,
                                                    hyp_L1_nodes, hyp_L2_nodes, hyp_L1_dropout_rate, hyp_L2_dropout_rate, hyp_batch_size, hyp_epochs, df_NN, 'Predictive parity')
                        df_NN_prp.replace(to_replace=[None], value='None', inplace=True)
                        df_MODEL_prp[4] = df_NN_prp
                        st.write(df_NN_prp)
                        csv = convert_df(df_NN_prp)

                        download_button_str = download_button(csv, "NN_pareto_prp.csv", 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                    if prE_ok:
                        st.subheader('NN - Predictive Equality')
                        df_NN_prE = SingleModelParetoNN(df_NN_prE, prE_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                    mean_test_scores_np, sem_test_scores_np,
                                                    hyp_L1_nodes, hyp_L2_nodes, hyp_L1_dropout_rate, hyp_L2_dropout_rate, hyp_batch_size, hyp_epochs, df_NN, 'Predictive equality')
                        df_NN_prE.replace(to_replace=[None], value='None', inplace=True)
                        df_MODEL_prE[4] = df_NN_prE
                        st.write(df_NN_prE)
                        csv = convert_df(df_NN_prE)
                        download_button_str = download_button(csv, "NN_pareto_prE.csv", 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                    if Eop_ok:
                        st.subheader('NN - Equal Opportunity')
                        df_NN_Eop = SingleModelParetoNN(df_NN_Eop, Eop_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                    mean_test_scores_np, sem_test_scores_np,
                                                    hyp_L1_nodes, hyp_L2_nodes, hyp_L1_dropout_rate, hyp_L2_dropout_rate, hyp_batch_size, hyp_epochs, df_NN, 'Equal opportunity')
                        df_NN_Eop.replace(to_replace=[None], value='None', inplace=True)
                        df_MODEL_Eop[4] = df_NN_Eop
                        st.write(df_NN_Eop)
                        csv = convert_df(df_NN_Eop)
                        download_button_str = download_button(csv, "NN_pareto_Eop.csv", 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                    if Acc_ok:
                        st.subheader('NN - Accuracy Equality')
                        df_NN_Acc = SingleModelParetoNN(df_NN_Acc, Acc_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                    mean_test_scores_np, sem_test_scores_np,
                                                    hyp_L1_nodes, hyp_L2_nodes, hyp_L1_dropout_rate, hyp_L2_dropout_rate, hyp_batch_size, hyp_epochs, df_NN, 'Accuracy equality')
                        df_NN_Acc.replace(to_replace=[None], value='None', inplace=True)
                        df_MODEL_Acc[4] = df_NN_Acc
                        st.write(df_NN_Acc)
                        csv = convert_df(df_NN_Acc)
                        download_button_str = download_button(csv, "NN_pareto_Acc.csv", 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                    if Eq1_ok:
                        st.subheader('NN - Equalized Odds')
                        df_NN_Eq1 = SingleModelParetoNN(df_NN_Eq1, Eq1_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                    mean_test_scores_np, sem_test_scores_np,
                                                    hyp_L1_nodes, hyp_L2_nodes, hyp_L1_dropout_rate, hyp_L2_dropout_rate, hyp_batch_size, hyp_epochs, df_NN, 'Equalized odds')
                        df_NN_Eq1.replace(to_replace=[None], value='None', inplace=True)
                        df_MODEL_Eq1[4] = df_NN_Eq1
                        st.write(df_NN_Eq1)
                        csv = convert_df(df_NN_Eq1)
                        download_button_str = download_button(csv, "NN_pareto_Eq1.csv", 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                    if Pty_ok:
                        st.subheader('NN - Statistical Parity')
                        df_NN_Pty = SingleModelParetoNN(df_NN_Pty, Pty_np, prp_np, sem_prp_np, prE_np, sem_prE_np, Eop_np, sem_Eop_np, Acc_np, sem_Acc_np, Eq1_np, sem_Eq1_np, Pty_np, sem_Pty_np,
                                                    mean_test_scores_np, sem_test_scores_np,
                                                    hyp_L1_nodes, hyp_L2_nodes, hyp_L1_dropout_rate, hyp_L2_dropout_rate, hyp_batch_size, hyp_epochs, df_NN, 'Statistical parity')
                        df_NN_Pty.replace(to_replace=[None], value='None', inplace=True)
                        df_MODEL_Pty[4] = df_NN_Pty
                        st.write(df_NN_Pty)
                        csv = convert_df(df_NN_Pty)
                        download_button_str = download_button(csv, "NN_pareto_Pty.csv", 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                    st.subheader('NN - Fairness Metrics Comparison')

                    if prp_ok:
                        prp_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_NN_prp['Accuracy'],
                                y=df_NN_prp['Predictive parity'],
                                name='Predictive parity',
                                marker=dict(size=6, color="red", symbol='circle',line=dict(width=2,
                                                                color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_NN_prp['L1_nodes'], df_NN_prp['L2_nodes'], df_NN_prp['L1_dropout_rate'], 
                                                        df_NN_prp['L2_dropout_rate'], df_NN_prp['batch_size'], df_NN_prp['epochs']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'L1_nodes: %{customdata[0]} <br>' +
                                'L2_nodes: %{customdata[1]} <br>' +
                                'L1_dropout_rate: %{customdata[2]} <br>' +
                                'L2_dropout_rate: %{customdata[3]} <br>' +
                                'batch_size: %{customdata[4]} <br>' +
                                'epochs: %{customdata[5]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_NN_prp['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_NN_prp['Predictive parity SEM'], visible=True),
                                )
                    else: prp_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                        
                    if prE_ok:
                        prE_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_NN_prE['Accuracy'],
                                y=df_NN_prE['Predictive equality'],
                                name='Predictive equality',
                                marker=dict(size=6, color="green", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_NN_prE['L1_nodes'], df_NN_prE['L2_nodes'], df_NN_prE['L1_dropout_rate'], 
                                                        df_NN_prE['L2_dropout_rate'], df_NN_prE['batch_size'], df_NN_prE['epochs']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'L1_nodes: %{customdata[0]} <br>' +
                                'L2_nodes: %{customdata[1]} <br>' +
                                'L1_dropout_rate: %{customdata[2]} <br>' +
                                'L2_dropout_rate: %{customdata[3]} <br>' +
                                'batch_size: %{customdata[4]} <br>' +
                                'epochs: %{customdata[5]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_NN_prE['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_NN_prE['Predictive equality SEM'], visible=True),
                            )
                    else: prE_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                        
                    if Eop_ok:
                        Eop_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_NN_Eop['Accuracy'],
                                y=df_NN_Eop['Equal opportunity'],
                                name='Equal opportunity',
                                marker=dict(size=6, color="gold", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_NN_Eop['L1_nodes'], df_NN_Eop['L2_nodes'], df_NN_Eop['L1_dropout_rate'], 
                                                        df_NN_Eop['L2_dropout_rate'], df_NN_Eop['batch_size'], df_NN_Eop['epochs']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'L1_nodes: %{customdata[0]} <br>' +
                                'L2_nodes: %{customdata[1]} <br>' +
                                'L1_dropout_rate: %{customdata[2]} <br>' +
                                'L2_dropout_rate: %{customdata[3]} <br>' +
                                'batch_size: %{customdata[4]} <br>' +
                                'epochs: %{customdata[5]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_NN_Eop['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_NN_Eop['Equal opportunity SEM'], visible=True),
                            )
                    else: Eop_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
                        
                    if Acc_ok:
                        Acc_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_NN_Acc['Accuracy'],
                                y=df_NN_Acc['Accuracy equality'],
                                name='Accuracy equality',
                                marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_NN_Acc['L1_nodes'], df_NN_Acc['L2_nodes'], df_NN_Acc['L1_dropout_rate'], 
                                                        df_NN_Acc['L2_dropout_rate'], df_NN_Acc['batch_size'], df_NN_Acc['epochs']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'L1_nodes: %{customdata[0]} <br>' +
                                'L2_nodes: %{customdata[1]} <br>' +
                                'L1_dropout_rate: %{customdata[2]} <br>' +
                                'L2_dropout_rate: %{customdata[3]} <br>' +
                                'batch_size: %{customdata[4]} <br>' +
                                'epochs: %{customdata[5]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_NN_Acc['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_NN_Acc['Accuracy equality SEM'], visible=True),
                            )
                    else: Acc_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                    if Eq1_ok:
                        Eq1_scatter = go.Scatter(
                                mode='lines+markers',
                                x=df_NN_Eq1['Accuracy'],
                                y=df_NN_Eq1['Equalized odds'],
                                name='Equalized odds',
                                marker=dict(size=6, color="black", symbol='circle',line=dict(width=2,
                                                            color='darkslategray')),
                                line=dict(width=2),
                                customdata = np.stack((df_NN_Eq1['L1_nodes'], df_NN_Eq1['L2_nodes'], df_NN_Eq1['L1_dropout_rate'], 
                                                        df_NN_Eq1['L2_dropout_rate'], df_NN_Eq1['batch_size'], df_NN_Eq1['epochs']), axis=-1),
                                hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                'Statistical parity: %{y:,.4f} <br>' + 
                                'L1_nodes: %{customdata[0]} <br>' +
                                'L2_nodes: %{customdata[1]} <br>' +
                                'L1_dropout_rate: %{customdata[2]} <br>' +
                                'L2_dropout_rate: %{customdata[3]} <br>' +
                                'batch_size: %{customdata[4]} <br>' +
                                'epochs: %{customdata[5]} <br>' +
                                '<extra>ok</extra>'),
                                error_x = dict(type='data', array=df_NN_Eq1['Accuracy SEM'], visible=True),
                                error_y = dict(type='data', array=df_NN_Eq1['Equalized odds SEM'], visible=True),
                                )
                    else: Eq1_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")


                    if Pty_ok:
                        Pty_scatter = go.Scatter(
                                    mode='lines+markers',
                                    x=df_NN_Pty['Accuracy'],
                                    y=df_NN_Pty['Statistical parity'],
                                    name='Statistical parity',
                                    marker=dict(size=6, color="blue", symbol='circle',line=dict(width=2,
                                                                color='darkslategray')),
                                    line=dict(width=2),
                                    customdata = np.stack((df_NN_Pty['L1_nodes'], df_NN_Pty['L2_nodes'], df_NN_Pty['L1_dropout_rate'], 
                                                        df_NN_Pty['L2_dropout_rate'], df_NN_Pty['batch_size'], df_NN_Pty['epochs']), axis=-1),
                                    hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
                                    'Statistical parity: %{y:,.4f} <br>' + 
                                    'L1_nodes: %{customdata[0]} <br>' +
                                    'L2_nodes: %{customdata[1]} <br>' +
                                    'L1_dropout_rate: %{customdata[2]} <br>' +
                                    'L2_dropout_rate: %{customdata[3]} <br>' +
                                    'batch_size: %{customdata[4]} <br>' +
                                    'epochs: %{customdata[5]} <br>' +
                                    '<extra>ok</extra>'),
                                    error_x = dict(type='data', array=df_NN_Pty['Accuracy SEM'], visible=True),
                                    error_y = dict(type='data', array=df_NN_Pty['Predictive parity SEM'], visible=True),
                        )
                    else: Pty_scatter = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")

                    data_legend = [prp_scatter, prE_scatter, Eop_scatter, Acc_scatter, Eq1_scatter, Pty_scatter]

                    fig = go.Figure(data=data_legend)

                    fig.update_yaxes(title='Fairness', title_font=dict(size=20), visible=True)
                    fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
                    fig.update_traces(line_shape="vh")
                    fig.update_layout(yaxis = dict(tickfont = dict(size=14)), xaxis = dict(tickfont = dict(size=15)))

                    st.plotly_chart(fig)

            ########################################################
            ###---------- Model Performance Comparison ----------###
            ########################################################

            st.subheader('Model Performance Comparison')

            if opt_method=='Grid Search':

                if prp_ok:
                    fig = SPF.SuperimposedPF(df_MODEL_prp, 'Predictive parity')
                    st.plotly_chart(fig)

                if prE_ok:
                    fig = SPF.SuperimposedPF(df_MODEL_prE, 'Predictive equality')
                    st.plotly_chart(fig)

                if Eop_ok:
                    fig = SPF.SuperimposedPF(df_MODEL_Eop, 'Equal opportunity')
                    st.plotly_chart(fig)

                if Acc_ok:
                    fig = SPF.SuperimposedPF(df_MODEL_Acc, 'Accuracy equality')
                    st.plotly_chart(fig)

                if Eq1_ok:
                    fig = SPF.SuperimposedPF(df_MODEL_Eq1, 'Equalized odds')
                    st.plotly_chart(fig)

                if Pty_ok:
                    fig = SPF.SuperimposedPF(df_MODEL_Pty, 'Statistical parity')
                    st.plotly_chart(fig)

                st.subheader('Multi-Model Pareto Fronts')
                
                if prp_ok:
                    df_cum_prp = pd.concat(df_MODEL_prp, ignore_index=True)
                if prE_ok:
                    df_cum_prE = pd.concat(df_MODEL_prE, ignore_index=True)
                if Eop_ok:
                    df_cum_Eop = pd.concat(df_MODEL_Eop, ignore_index=True)
                if Acc_ok:
                    df_cum_Acc = pd.concat(df_MODEL_Acc, ignore_index=True)
                if Eq1_ok:
                    df_cum_Eq1 = pd.concat(df_MODEL_Eq1, ignore_index=True)
                if Pty_ok:
                    df_cum_Pty = pd.concat(df_MODEL_Pty, ignore_index=True)

                if prp_ok:
                    df_cum_prp = pt.pareto_frontier(Xs=df_cum_prp['Accuracy'], Ys=df_cum_prp['Predictive parity'], name='Predictive parity', maxX=True, maxY=True)
                    fig = mmpf.MMPF(df_cum_prp, df_MODEL, 'Predictive parity')
                    st.plotly_chart(fig)

                if prE_ok:
                    df_cum_prE = pt.pareto_frontier(Xs=df_cum_prE['Accuracy'], Ys=df_cum_prE['Predictive equality'], name='Predictive equality', maxX=True, maxY=True)
                    fig = mmpf.MMPF(df_cum_prE, df_MODEL, 'Predictive equality')
                    st.plotly_chart(fig)

                if Eop_ok:
                    df_cum_Eop = pt.pareto_frontier(Xs=df_cum_Eop['Accuracy'], Ys=df_cum_Eop['Equal opportunity'], name='Equal opportunity', maxX=True, maxY=True)
                    fig = mmpf.MMPF(df_cum_Eop, df_MODEL, 'Equal opportunity')
                    st.plotly_chart(fig)

                if Acc_ok:
                    df_cum_Acc = pt.pareto_frontier(Xs=df_cum_Acc['Accuracy'], Ys=df_cum_Acc['Accuracy equality'], name='Accuracy equality', maxX=True, maxY=True)
                    fig = mmpf.MMPF(df_cum_Acc, df_MODEL, 'Accuracy equality')
                    st.plotly_chart(fig)

                if Eq1_ok:
                    df_cum_Eq1 = pt.pareto_frontier(Xs=df_cum_Eq1['Accuracy'], Ys=df_cum_Eq1['Equalized odds'], name='Equalized odds', maxX=True, maxY=True)
                    fig = mmpf.MMPF(df_cum_Eq1, df_MODEL, 'Equalized odds')
                    st.plotly_chart(fig)

                if Pty_ok:
                    df_cum_Pty = pt.pareto_frontier(Xs=df_cum_Pty['Accuracy'], Ys=df_cum_Pty['Statistical parity'], name='Statistical parity', maxX=True, maxY=True)
                    fig = mmpf.MMPF(df_cum_Pty, df_MODEL, 'Statistical parity')
                    st.plotly_chart(fig)

            else:
                if prp_ok:
                    df_cum_prp = pd.concat(df_MODEL_prp, ignore_index=True)
                if prE_ok:
                    df_cum_prE = pd.concat(df_MODEL_prE, ignore_index=True)
                if Eop_ok:
                    df_cum_Eop = pd.concat(df_MODEL_Eop, ignore_index=True)
                if Acc_ok:
                    df_cum_Acc = pd.concat(df_MODEL_Acc, ignore_index=True)
                if Eq1_ok:
                    df_cum_Eq1 = pd.concat(df_MODEL_Eq1, ignore_index=True)
                if Pty_ok:
                    df_cum_Pty = pd.concat(df_MODEL_Pty, ignore_index=True)

                st.subheader('Multi-Model Pareto Fronts')

                if prp_ok:
                    df_cum_prp = pt.pareto_frontier(Xs=df_cum_prp['accuracy'], Ys=df_cum_prp['predictive parity'], name='predictive parity', maxX=True, maxY=True)
                    fig = mmpf_bo.MMPFBO(df_cum_prp, df_MODEL_prp, 'predictive parity')
                    st.plotly_chart(fig)

                if prE_ok:
                    df_cum_prE = pt.pareto_frontier(Xs=df_cum_prE['accuracy'], Ys=df_cum_prE['predictive equality'], name='predictive equality', maxX=True, maxY=True)
                    fig = mmpf_bo.MMPFBO(df_cum_prE, df_MODEL_prE, 'predictive equality')
                    st.plotly_chart(fig)

                if Eop_ok:
                    df_cum_Eop = pt.pareto_frontier(Xs=df_cum_Eop['accuracy'], Ys=df_cum_Eop['equal opportunity'], name='equal opportunity', maxX=True, maxY=True)
                    fig = mmpf_bo.MMPFBO(df_cum_Eop, df_MODEL_Eop, 'equal opportunity')
                    st.plotly_chart(fig)

                if Acc_ok:
                    df_cum_Acc = pt.pareto_frontier(Xs=df_cum_Acc['accuracy'], Ys=df_cum_Acc['accuracy equality'], name='accuracy equality', maxX=True, maxY=True)
                    fig = mmpf_bo.MMPFBO(df_cum_Acc, df_MODEL_Acc, 'accuracy equality')
                    st.plotly_chart(fig)

                if Eq1_ok:
                    df_cum_Eq1 = pt.pareto_frontier(Xs=df_cum_Eq1['accuracy'], Ys=df_cum_Eq1['equalized odds'], name='equalized odds', maxX=True, maxY=True)
                    fig = mmpf_bo.MMPFBO(df_cum_Eq1, df_MODEL_Eq1, 'equalized odds')
                    st.plotly_chart(fig)

                if Pty_ok:
                    df_cum_Pty = pt.pareto_frontier(Xs=df_cum_Pty['accuracy'], Ys=df_cum_Pty['statistical parity'], name='statistical parity', maxX=True, maxY=True)
                    fig = mmpf_bo.MMPFBO(df_cum_Pty, df_MODEL_Pty, 'statistical parity')
                    st.plotly_chart(fig)



        else:
            st.write('The configuration is incomplete!')
    
