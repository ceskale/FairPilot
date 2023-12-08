# Standard libraries
import os
import json
import pickle
import uuid
import re
from collections import deque
from functools import reduce, partial
from itertools import product

# Data handling and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning: scikit-learn
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

# FairXplainer
# from fairxplainer.fair_explainer import plot as fif_plot
# from fairxplainer.fair_explainer import FairXplainer

# Configuration and optimization
from openbox import space as sp
from openbox import Optimizer
import ConfigSpace as cs
from ConfigSpace.util import get_one_exchange_neighbourhood  # for hyperparameter combinations
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition, AndConjunction
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenInClause

# Miscellaneous
import streamlit as st
import base64

def encode_object(obj, pickle_it=False):
    """
    Encodes an object to bytes, handling several common types.
    
    Params:
    ------
    obj: Object to be encoded.
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (bytes): encoded object
    """
    if pickle_it:
        return pickle.dumps(obj)

    if isinstance(obj, bytes):
        return obj

    if isinstance(obj, pd.DataFrame):
        return obj.to_csv(index=False).encode()

    # Assume it's JSON-able for everything else
    return json.dumps(obj).encode()


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download: The object to be downloaded.
    download_filename (str): filename and extension of file.
    button_text (str): Text to display on download button.
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download
    """
    try:
        encoded_object = encode_object(object_to_download, pickle_it)
        b64 = base64.b64encode(encoded_object).decode()

    except Exception as e:
        st.write("Error encoding object:", e)
        return None

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub(r'\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(8, 232, 150);
                text-align:center;
                color:#000000;
                padding: 8px 20px;
                position: relative;
                text-decoration: none;
                border-radius: 6px;
                border: 1px solid rgb(200, 200, 200);
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

    dl_link = f'{custom_css}<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

#################################
#################################
###--------- SIDEBAR ---------###
#################################
#################################
st.sidebar.image("logo.png", use_column_width=True)

# Create a sidebar header
st.sidebar.header('FairPilot Settings')

# Add a subheader for Optimization Algorithm
st.sidebar.subheader('Optimization Algorithm')

# Define optimization methods and their descriptions
optimization_methods = {
    'Grid Search': "Build a custom grid and try out every possible HP combination. Allows for a better interpretation of Hyperparameter's influence on Fairness.",
    'MOBO': "The optimization is performed fully automatically in a smarter way. Outputs results are closer to the optimal values."
}

# Create a selectbox for the optimization methods
opt_method = st.sidebar.selectbox(
    'Choose the appropriate Optimization Algorithm',
    options=list(optimization_methods.keys()),
    format_func=lambda method: f"{method} - {optimization_methods[method]}"
)

def adjusted_range(start, end, step, min_val=None, max_val=None):
    values = np.arange(start, end + step, step)
    if min_val is not None and values[0] < min_val:
        values[0] = min_val
    if max_val is not None and values[-1] > max_val:
        values[-1] = max_val
    return values

# Decision Tree Classifier
checkDT = st.sidebar.checkbox('Decision Tree Classifier')
if opt_method == 'Grid Search' and checkDT:
    with st.sidebar.expander('**Set HP space:**'):
        criterion = st.multiselect('Criterion', ['gini', 'entropy'], ['gini'])
        if not criterion:
            st.warning("Please select at least one criterion.")
            criterion = ['gini']

        max_depth_step = st.number_input('Max depth - Step size',1,5)
        max_depth = adjusted_range(*st.slider('Max depth', 5, 100, [5, 25], max_depth_step), max_depth_step, max_val=100)
        # For slider widgets, you don't need to add a check since a range is always selected
        if max_depth[0] == max_depth[-1]:
            st.warning("Please select a valid range for Max depth.")
            max_depth = [5, 25]

        max_features = np.array(st.multiselect('Max-features', ['sqrt', 'log2'], ['sqrt']))
        if max_features.size == 0:
            st.warning("Please select at least one value for Max-features.")
            max_features = ['sqrt']

        min_samples_split_step = st.number_input('Min-samples split - Step size',1,5)
        min_samples_split = adjusted_range(*st.slider('Min-samples split', 2, 20, [2, 10], min_samples_split_step), min_samples_split_step, max_val=20)
        # Again, the slider will always have a value or range
        if min_samples_split[0] == min_samples_split[-1]:
            st.warning("Please select a valid range for min samples split.")
            min_samples_split = [2, 10]

        min_samples_leaf_step = st.number_input('Min-samples leaf - Step size',1,5)
        min_samples_leaf = adjusted_range(*st.slider('Min-samples leaf', 1, 20, [5, 10], min_samples_leaf_step), min_samples_leaf_step, max_val=20)
        # The slider will always return a value or range
        if min_samples_leaf[0] == min_samples_leaf[-1]:
            st.warning("Please select a valid range for min samples leaf.")
            min_samples_leaf = [5, 10]

# Support Vector Classifier
checkSVC = st.sidebar.checkbox('Support Vector Classifier')
if opt_method == 'Grid Search' and checkSVC:
    with st.sidebar.expander('**Set HP space:**'):
        kernels = st.multiselect('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], default='poly')
        if not kernels:
            st.warning("Please select at least one kernel.")
            kernels = ['poly']

        Cs = np.array(list(map(float, st.multiselect('C ', ['1000', '100', '10', '1', '0.1', '0.01', '0.001'], ['1']))))
        if Cs.size == 0:
            st.warning("Please select at least one value for C.")
            Cs = [1.0]

        shrinkings = st.multiselect('shrinking', [True, False], default=True)
        if not shrinkings:
            st.warning("Please select a value for 'shrinking'.")
            shrinkings = [True]

        degrees = st.slider('Select a range of degrees', 1, 5, (1, 5), 1)
        if degrees[0] == degrees[1]:
            st.warning("Please select a valid range for degrees.")
            degrees = (1, 5)
        # For slider widgets, you don't need to add a check since a range is always selected

        coefs = st.slider('coef', 1, 10, (1, 3), 1)
        # Similarly, the 'coefs' slider will always have a value or range
        if coefs[0] == coefs[1]:
            st.warning("Please select a valid range for coef.")
            coefs = (1, 3)

        gamma_values = np.array(list(map(float, st.multiselect('gamma_value', ['1000', '100', '10', '1', '0.1', '0.01', '0.001'], ['1']))))
        if gamma_values.size==0:
            st.warning("Please select at least one value for gamma.")
            gamma_values = [1.0]


# Logistic Regression
checkLR = st.sidebar.checkbox('Logistic Regression')
if opt_method == 'Grid Search' and checkLR:
    with st.sidebar.expander('**Set HP space:**'):
        LR_penalties = st.multiselect('Penalty', ['none', 'l2'], ['none'])
        if not LR_penalties:
            st.warning("Please select at least one penalty.")
            LR_penalties = ['none']

        LR_Cs = np.array(list(map(float, st.multiselect('C', ['1000', '100', '10', '1', '0.1', '0.01', '0.001'], ['1']))))
        if LR_Cs.size == 0:
            st.warning("Please select at least one value for C.")
            LR_Cs = [1.0]

        LR_fit_intercept = st.multiselect('Fit intercept', [True, False], [True])
        if not LR_fit_intercept:
            st.warning("Please select a value for 'Fit intercept'.")
            LR_fit_intercept = [True]


# Random Forest Classifier
checkRF = st.sidebar.checkbox('Random Forest Classifier')
if opt_method == 'Grid Search' and checkRF:
    with st.sidebar.expander('**Set HP space:**'):
        bootstrap = st.multiselect('Bootstrap', [True, False], [True])
        if not bootstrap:
            st.warning("Please select at least one bootstrap option.")
            bootstrap = [True]

        max_samples = np.array(list(map(float, st.multiselect('Max bootstrap samples', ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], ['1.0']))))
        if max_samples.size == 0:
            st.warning("Please select at least one value for Max bootstrap samples.")
            max_samples = [1.0]

        RF_criterion = st.multiselect('Criterion ', ['gini', 'entropy'], ['gini'])
        if not RF_criterion:
            st.warning("Please select at least one criterion for Random Forest.")
            RF_criterion = ['gini']

        RF_max_depth_step = st.number_input('Max depth - Step size ', 1, 5)
        RF_max_depth = adjusted_range(*st.slider('Max depth ', 2, 100, [5, 25], RF_max_depth_step), RF_max_depth_step, max_val=100)
        if RF_max_depth[0] == RF_max_depth[-1]:
            st.warning("Please select a valid range for Max depth.")
            RF_max_depth = [5, 25]

        RF_max_features = st.multiselect('Max-features ', ['sqrt', 'log2'], ['sqrt'])
        if not RF_max_features:
            st.warning("Please select at least one value for Max-features.")
            RF_max_features = ['sqrt']

        RF_min_samples_split_step = st.number_input('Min-samples split - Step size ', 1, 5)
        RF_min_samples_split = adjusted_range(*st.slider('Min-samples split ', 2, 20, [2, 10], RF_min_samples_split_step), RF_min_samples_split_step, max_val=20)
        if RF_min_samples_split[0] == RF_min_samples_split[-1]:
            st.warning("Please select a valid range for min samples split.")
            RF_min_samples_split = [2, 10]

        RF_min_samples_leaf_step = st.number_input('Min-samples leaf - Step size ', 1, 5)
        RF_min_samples_leaf = adjusted_range(*st.slider('Min-samples leaf ', 1, 20, [5, 10], RF_min_samples_leaf_step), RF_min_samples_leaf_step, max_val=20)
        if RF_min_samples_leaf[0] == RF_min_samples_leaf[-1]:
            st.warning("Please select a valid range for min samples leaf.")
            RF_min_samples_leaf = [5, 10]


st.sidebar.write('---')

st.sidebar.subheader('Fairness Metrics')
fairness_metrics_info = {
    'Predictive Parity': 'Positive outcome probability should be same for both classes.',
    'Predictive Equality': 'False positive fraction should be the same for both classes.',
    'Equal Opportunity': 'False negative fraction should be the same for both classes.',
    'Accuracy Equality': 'Prediction accuracy should be the same for both classes.',
    'Equalized Odds': 'False positive AND true positive fractions should be the same for both classes.',
    'Statistical Parity': 'Positive predictions ratio should be the same regardless of the class.'
}

metrics_checkboxes = {
    'Predictive Parity': st.sidebar.checkbox('Predictive Parity', help=fairness_metrics_info['Predictive Parity']),
    'Predictive Equality': st.sidebar.checkbox('Predictive Equality', help=fairness_metrics_info['Predictive Equality']),
    'Equal Opportunity': st.sidebar.checkbox('Equal Opportunity', help=fairness_metrics_info['Equal Opportunity']),
    'Accuracy Equality': st.sidebar.checkbox('Accuracy Equality', help=fairness_metrics_info['Accuracy Equality']),
    'Equalized Odds': st.sidebar.checkbox('Equalized Odds', help=fairness_metrics_info['Equalized Odds']),
    'Statistical Parity': st.sidebar.checkbox('Statistical Parity', help=fairness_metrics_info['Statistical Parity'])
}

metrics_to_include = {
    'Predictive parity': metrics_checkboxes['Predictive Parity'],
    'Predictive equality': metrics_checkboxes['Predictive Equality'],
    'Equal opportunity': metrics_checkboxes['Equal Opportunity'],
    'Accuracy equality': metrics_checkboxes['Accuracy Equality'],
    'Equalized odds': metrics_checkboxes['Equalized Odds'],
    'Statistical parity': metrics_checkboxes['Statistical Parity']
}

st.sidebar.write('---')

st.sidebar.subheader('Advanced Settings')
n_seeds = st.sidebar.slider('**Number of Repetitions**', 2, 20, 5, 1)
scl_type = st.sidebar.selectbox('**Data Scaling**', ('None', 'Standardization'))
imp_type = st.sidebar.selectbox('**Data Imputation**',('Median', 'Mean'))
checkDUMMY = st.sidebar.selectbox('**Generate categorical variables**',('No', 'Yes'))


# Decision Tree Classifier
if opt_method == 'Grid Search' and checkDT:
    # Hyperparameter definitions
    cs_dtc = ConfigurationSpace()

    criterion_hp = CategoricalHyperparameter('criterion', criterion)
    max_depth_hp = CategoricalHyperparameter('max_depth', max_depth)
    max_features_hp = CategoricalHyperparameter('max_features', max_features)
    min_samples_split_hp = CategoricalHyperparameter('min_samples_split', min_samples_split)
    min_samples_leaf_hp = CategoricalHyperparameter('min_samples_leaf', min_samples_leaf)

    # Add hyperparameters to config space
    cs_dtc.add_hyperparameters([criterion_hp, max_depth_hp, max_features_hp, min_samples_split_hp, min_samples_leaf_hp])

# Support Vector Classifier
if opt_method == 'Grid Search' and checkSVC:
    cs_svc = ConfigurationSpace()

    kernel = CategoricalHyperparameter('kernel', kernels)
    C_hp = CategoricalHyperparameter('C', Cs)
    shrinking = CategoricalHyperparameter('shrinking', shrinkings)

    if 'poly' in kernels:
        degree = UniformIntegerHyperparameter('degree', min(degrees), max(degrees))
        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    else:
        use_degree = InCondition(child=C_hp, parent=kernel, values=[])
        degree = CategoricalHyperparameter('degree', [1])

    if 'poly' in kernels:
        coef = CategoricalHyperparameter('coef0', coefs)
        use_coef = InCondition(child=coef, parent=kernel, values=["poly"])
    elif 'sigmoid' in kernels:
        coef = CategoricalHyperparameter('coef0', coefs)
        use_coef = InCondition(child=coef, parent=kernel, values=["sigmoid"])
    elif 'poly' in kernels and 'sigmoid' in kernels:
        coef = CategoricalHyperparameter('coef0', coefs)
        use_coef = InCondition(child=coef, parent=kernel, values=["sigmoid", "poly"])
    else:
        use_coef = InCondition(child=C_hp, parent=kernel, values=[])
        coef = CategoricalHyperparameter('coef0', [0.0])

    if 'poly' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=gamma, parent=kernel, values=["poly"])
    elif 'sigmoid' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=gamma, parent=kernel, values=["sigmoid"])
    elif 'rbf' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf"])
    elif 'poly' in kernels and 'sigmoid' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
    elif 'poly' in kernels and 'rbf' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=coef, parent=kernel, values=["poly", "rbf"])
    elif 'sigmoid' in kernels and 'rbf' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=coef, parent=kernel, values=["sigmoid", "rbf"])
    elif 'sigmoid' in kernels and 'rbf' in kernels and 'poly' in kernels:
        gamma = CategoricalHyperparameter('gamma', gamma_values)
        use_gamma = InCondition(child=coef, parent=kernel, values=["sigmoid", "rbf", "poly"])
    else:
        gamma = CategoricalHyperparameter('gamma', ['auto'])
        use_gamma = InCondition(child=C_hp, parent=kernel, values=[])

    # Add hyperparameters and conditions to our configspace
    cs_svc.add_hyperparameters([kernel, C_hp, shrinking, degree, coef, gamma])
    cs_svc.add_conditions([use_degree, use_coef, use_gamma])

# Logistic Regression
if opt_method == 'Grid Search' and checkLR:
    cs_lr = ConfigurationSpace()

    # Define hyperparameters
    LR_C_hp = CategoricalHyperparameter('C', LR_Cs)
    LR_fit_intercept_hp = CategoricalHyperparameter('fit_intercept', LR_fit_intercept)
    LR_penalty_hp = CategoricalHyperparameter('penalty', LR_penalties)
    
    # Add other hyperparameters
    cs_lr.add_hyperparameters([LR_penalty_hp, LR_C_hp, LR_fit_intercept_hp,])

# Random Forest Classifier
if opt_method == 'Grid Search' and checkRF:
    cs_rf = ConfigurationSpace()

    bootstrap_hp = CategoricalHyperparameter('bootstrap', bootstrap)
    RF_criterion_hp = CategoricalHyperparameter('criterion', RF_criterion)
    RF_max_depth_hp = CategoricalHyperparameter('max_depth', RF_max_depth)
    RF_max_features_hp = CategoricalHyperparameter('max_features', RF_max_features)
    RF_min_samples_split_hp = CategoricalHyperparameter('min_samples_split', RF_min_samples_split)
    RF_min_samples_leaf_hp = CategoricalHyperparameter('min_samples_leaf', RF_min_samples_leaf)

    cs_rf.add_hyperparameters([bootstrap_hp, RF_criterion_hp, RF_max_depth_hp, RF_max_features_hp, RF_min_samples_split_hp, RF_min_samples_leaf_hp])

    if True in bootstrap:
        max_samples_hp = CategoricalHyperparameter('max_samples', max_samples)
        use_max_samples = InCondition(child=max_samples_hp, parent=bootstrap_hp, values=[True])

        cs_rf.add_hyperparameters([max_samples_hp])
        cs_rf.add_conditions([use_max_samples])


###################################
###################################
###--------- MAIN MENU ---------###
###################################
###################################


st.write("# FairPilot")
#st.caption('By Francesco Di Carlo - OPEX Lab at UIC')

# Provide a description for the file uploader
st.write('Welcome to FairPilot. You can perform Multi-Objective optimization of four different supervised learning methods. The results show the accuracy-fairness Pareto Fronts for numerous fairness definitions.')

# Create a subheader for data upload
st.subheader('Upload Your Dataset')

# Implement the file uploader
file = st.file_uploader('', type=['xlsx', 'csv'])

if file is not None:
    if file.type == "text/csv":
        df = pd.read_csv(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
    else:
        st.warning("Please upload a file in CSV or XLSX format.")
        file = None  # Invalidate the file so the following code doesn't run

    # Dummy Variable Generation
    if checkDUMMY == 'Yes':
        df = pd.get_dummies(df, drop_first=True)

    def is_binary(series):
        return series.nunique() == 2

    binary_columns = [col for col in df.columns if is_binary(df[col])]

    if not binary_columns:
        st.warning("No binary columns found in the dataset.")
    else:
        y_name = st.selectbox('Choose output variable', binary_columns)
        X_sen = st.selectbox('Choose sensitive attribute', [col for col in binary_columns if col != y_name])

        if y_name and X_sen:
            available_columns = [col for col in df.columns if col not in [y_name, X_sen]]
            proxies = st.multiselect('Choose proxies (up to 5)', available_columns, default=[])

            st.write('---')

            st.subheader('Dataset Exploration')

            # Total instances of positive and negative outcomes for y_name
            positive_outcomes = df[y_name].sum()
            negative_outcomes = len(df) - positive_outcomes
            
            # Total instances of x_sen=0 and 1
            x_sen_0_count = len(df[df[X_sen] == 0])
            x_sen_1_count = len(df) - x_sen_0_count
            
            # Ratio of y_name=1 when x_sen=1 and the ratio of y_name=1 when x_sen=0
            ratio_y1_given_x_sen1 = df[(df[y_name] == 1) & (df[X_sen] == 1)].shape[0] / x_sen_1_count
            ratio_y1_given_x_sen0 = df[(df[y_name] == 1) & (df[X_sen] == 0)].shape[0] / x_sen_0_count
            
            # Create a DataFrame to display the results in a tabular format
            exploration_data = {
                'Description': [
                    f"Total instances of positive outcomes ({y_name}=1)",
                    f"Total instances of negative outcomes ({y_name}=0)",
                    f"Total instances where {X_sen}=0",
                    f"Total instances where {X_sen}=1",
                    f"Ratio of positive outcomes when {X_sen}=1",
                    f"Ratio of positive outcomes when {X_sen}=0"
                ],
                'Value': [
                    positive_outcomes,
                    negative_outcomes,
                    x_sen_0_count,
                    x_sen_1_count,
                    f"{ratio_y1_given_x_sen1:.2f}",
                    f"{ratio_y1_given_x_sen0:.2f}"
                ]
            }
            
            exploration_df = pd.DataFrame(exploration_data)
            
            # Display the table
            st.table(exploration_df)
            st.write('---')

            if len(proxies) > 5:
                st.warning("Please select up to 5 proxies only.")
                proxies = proxies[:5]

            if proxies:
                # Visualization code (histograms and boxplots)
                fig = make_subplots(rows=2, cols=len(proxies), subplot_titles=[f'Histogram for {proxy}' for proxy in proxies] + [f'Boxplot for {proxy}' for proxy in proxies])

                # Define the shade of green
                colors = ['#66c2a5', '#238b45']

                for idx, proxy in enumerate(proxies):
                    # Histograms
                    hist_0 = go.Histogram(x=df[df[X_sen] == 0][proxy], name=f'{X_sen}=0', marker_color=colors[0], histnorm='percent')
                    hist_1 = go.Histogram(x=df[df[X_sen] == 1][proxy], name=f'{X_sen}=1', marker_color=colors[1], histnorm='percent')
                    
                    fig.add_trace(hist_0, row=1, col=idx+1)
                    fig.add_trace(hist_1, row=1, col=idx+1)

                    # Boxplots
                    box_0 = go.Box(y=df[df[X_sen] == 0][proxy], name=f'{X_sen}=0', marker_color=colors[0])
                    box_1 = go.Box(y=df[df[X_sen] == 1][proxy], name=f'{X_sen}=1', marker_color=colors[1])
                    
                    fig.add_trace(box_0, row=2, col=idx+1)
                    fig.add_trace(box_1, row=2, col=idx+1)

                # Update the layout
                fig.update_layout(barmode='group', bargap=0.1, bargroupgap=0.1, title_text="Proxies Analysis", showlegend=False)

                # Display the combined figure in Streamlit
                st.plotly_chart(fig)

                # Imputation
                if imp_type == 'Median':
                    medians = df.select_dtypes(include=[np.number]).median()
                    df = df.fillna(medians)
                elif imp_type == 'Mean':
                    means = df.select_dtypes(include=[np.number]).mean()
                    df = df.fillna(means)

                # Scaling
                if scl_type == 'Standardization':
                    scaler = MinMaxScaler()
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                # Extract y_name as the target variable and drop it from df
                df = df.dropna(subset=[y_name])
                y = df[y_name]
                X_scaled = df.drop(columns=[y_name])

                st.subheader('Processed Dataset')
                st.write(df.head(5))  # Display first five rows of the processed dataset

m = st.markdown("""
    <style>
        /* Base styling for button */
        .stDownloadButton button, div.stButton > button:first-child {
            background-color: rgb(8, 232, 150);
            color: #000000;
            padding-left: 20px;
            padding-right: 20px;
            border-radius: 5px;
            border: 1px solid rgb(200, 200, 200);
            text-align: center;
        }

        /* Hover effects for button */
        .stDownloadButton button:hover, div.stButton > button:hover {
            color: rgb(8, 130, 50);
            border-color: rgb(8, 130, 50);
            background-color: rgb(8, 232, 135);
        }

        /* Focus effects for button */
        .stDownloadButton button:focus, div.stButton > button:focus {
            background-color: rgb(8, 150, 135);
            color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

button = st.button('**START**')
if button and file is None:
    st.warning('Upload a file!')
elif button and file is not None:

    st.write('The color in the graphs represents the gradient for max_depth in tree-based methods and C in the other two.')

    #affirmative_action = False # set to True to see the effect of affirmative action
    #clf = DecisionTreeClassifier()
    #clf.fit(X.values, y.values)

    #verbose = False
    #fairXplainer = FairXplainer(classifier=clf, dataset=X_scaled, sensitive_features=['Privileged'], verbose=True)
    #fairXplainer.compute(maxorder=1, spline_intervals=6, verbose=False)

    #explanation_result = fairXplainer.get_top_k_weights(k=10)

    #plt = fif_plot(explanation_result, draw_waterfall=True, labelsize=18, fontsize=20, figure_size=(6, 2.7), title="", text_x_pad=0.05, text_y_pad=0.2, result_y_location=0.6, result_x_pad=0.02,
    #        x_label=r"Statistical parity", delete_zero_weights=True)
    #plt.show()
    #print("Exact statistical parity:", fairXplainer.statistical_parity_sample())

    def convert_df(df):
        """Converts a Pandas DataFrame into a base64-encoded string for CSV format."""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert bytes to string
        return b64

    def fairness(y1_pred, y1, y0_pred, y0, metrics_to_include):
        """Compute fairness metrics based on the predictions and true labels."""

        # Confusion matrices
        TN1, FP1, FN1, TP1 = confusion_matrix(y1, y1_pred).ravel()
        TN0, FP0, FN0, TP0 = confusion_matrix(y0, y0_pred).ravel()
        
        tot1 = TN1 + FP1 + FN1 + TP1
        tot0 = TN0 + FP0 + FN0 + TP0

        metrics = {}

        # Predictive parity
        if metrics_to_include.get('Predictive parity'):
            metrics['Predictive parity'] = abs(TP1/(TP1+FP1) - TP0/(TP0+FP0))

        # Predictive equality
        if metrics_to_include.get('Predictive equality'):
            metrics['Predictive equality'] = abs(FP1/(TN1+FP1) - FP0/(TN0+FP0))

        # Equal opportunity
        if metrics_to_include.get('Equal opportunity'):
            metrics['Equal opportunity'] = abs(FN1/(TP1+FN1) - FN0/(TP0+FN0))

        # Accuracy equality
        if metrics_to_include.get('Accuracy equality'):
            metrics['Accuracy equality'] = abs((TN1+TP1)/tot1 - (TN0+TP0)/tot0)

        # Equalized odds
        if metrics_to_include.get('Equalized odds'):
            TPR_diff = abs(TP1/(TP1+FN1) - TP0/(TP0+FN0))
            FPR_diff = abs(FP1/(TN1+FP1) - FP0/(TN0+FP0))
            metrics['Equalized odds'] = 0.5 * (TPR_diff + FPR_diff)

        # Statistical parity
        if metrics_to_include.get('Statistical parity'):
            metrics['Statistical parity'] = abs((FP1 + TP1)/tot1 - (FP0 + TP0)/tot0)
        
        return metrics

    def plotly_parallel_coordinates(df):
        # Convert categorical columns to type 'category' if not already
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype('category')
    
        # Create the parallel coordinates plot with color scale from red to green
        fig = px.parallel_coordinates(df, color="Accuracy",
                                      color_continuous_scale=['red', 'blue'],
                                      labels={col: col.replace('_', ' ') for col in df.columns})
    
        fig.update_layout(
        margin=dict(l=20,) #r=50, t=50, b=50),
        )

        return fig

    def FairGridCV(X, y, sensitive_attribute, classifier, n_folds, hyperparameters, metrics_to_include):
    
        test_scores = []
        fairness_metrics = []
        results = []

        #total_combinations = reduce(lambda x, y: x*y, [len(hp.choices) if isinstance(hp, cs.hyperparameters.CategoricalHyperparameter) else 1 for hp in hyperparameters.values()])
        
        # Extract choices or range for each hyperparameter
        def get_hyperparameter_values(hp):
            if isinstance(hp, cs.hyperparameters.CategoricalHyperparameter):
                return hp.choices
            elif isinstance(hp, cs.hyperparameters.UniformIntegerHyperparameter):
                return list(range(hp.lower, hp.upper+1))
            else:
                return [hp]
        
        # Prepare the progress bar
        progress = st.progress(0, 'Validating your models...')

        def get_num_choices(hp):
            if isinstance(hp, cs.hyperparameters.CategoricalHyperparameter):
                return len(hp.choices)
            elif isinstance(hp, cs.hyperparameters.UniformIntegerHyperparameter):
                return hp.upper - hp.lower + 1
            else:
                return 1
            
        total_combinations = reduce(lambda x, y: x*y, [get_num_choices(hp) for hp in hyperparameters.values()])

        # Iterate through hyperparameter combinations
        for i, hyperparams in enumerate(product(*[get_hyperparameter_values(hp) for hp in hyperparameters.values()])):
            
            hyperparams_dict = dict(zip(hyperparameters.keys(), hyperparams))

            # Check if 'bootstrap' is a hyperparameter and set 'max_samples' accordingly
            if 'bootstrap' in hyperparams_dict:
                if not hyperparams_dict['bootstrap']:
                    hyperparams_dict['max_samples'] = None
            
            for train_index, test_index in StratifiedKFold(n_splits=n_folds).split(X, y):
                
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]

                # Locate privileged and unprivileged groups
                X1_test = X_test.loc[(sensitive_attribute > 0)]
                X0_test = X_test.loc[(sensitive_attribute == 0)]
                y1_test = y_test.loc[(sensitive_attribute > 0)]
                y0_test = y_test.loc[(sensitive_attribute == 0)]

                # Fit the classifier
                clf = classifier(**hyperparams_dict)
                clf.fit(X_train, y_train)

                test_scores.append(clf.score(X_test, y_test))

                # Predict the labels for the test set
                y1_pred = clf.predict(X1_test)
                y0_pred = clf.predict(X0_test)

                # Calculate fairness metrics
                metrics = fairness(y1_pred, y1_test, y0_pred, y0_test, metrics_to_include)
                fairness_metrics.append(metrics)
            
            accuracy = sum(test_scores) / len(test_scores)
            
            # Construct the result dictionary
            result = {'Accuracy': accuracy}
            for metric_key, metric_value in metrics_to_include.items():
                if metric_value:
                    result[metric_key] = 1 - np.mean([metrics[metric_key] for metrics in fairness_metrics])

            
            test_scores = []
            fairness_metrics = []

            # Append hyperparameters to results, converting categorical parameters to string
            for hp_key, hp_value in hyperparams_dict.items():
                result[hp_key] = str(hp_value) if isinstance(hp_value, (list, tuple)) else hp_value
    
            # Append to results
            results.append(result)

            # Append hyperparameters to results
          #  result.update(hyperparams_dict)

            # Append to results
          #  results.append(result)

            # Update the progress bar
            progress.progress(min((i + 1) / total_combinations, 1.0))

        st.balloons()
        df = pd.DataFrame(results)
        # Convert categorical columns to type 'category'
        for col in hyperparameters.keys():
            if isinstance(hyperparameters[col], cs.hyperparameters.CategoricalHyperparameter):
                df[col] = df[col].astype('category')
        return df
    
    def pareto_frontier(df_fairgrid, metric_name, maxX=True, maxY=True):
        '''Pareto frontier selection process for potential trade-offs'''
        # Combine the two lists into a list of points and sort based on X values
        points = sorted([(index, row["Accuracy"], row[metric_name]) for index, row in df_fairgrid.iterrows()],
                        key=lambda x: x[1], reverse=not maxX)
        
        pareto_front = deque()
        pareto_front.append(points[0])
        
        for index, x, y in points:
           if maxY:
               while pareto_front and y >= pareto_front[-1][2]:
                   pareto_front.pop()
           else:
               while pareto_front and y <= pareto_front[-1][2]:
                   pareto_front.pop()
           pareto_front.append((index, x, y))

        # Create a DataFrame from Pareto front points
        pareto_indices, pf_X, pf_Y = zip(*pareto_front)
        pareto_df = df_fairgrid.loc[list(pareto_indices)].copy()
        pareto_df["Accuracy"] = pf_X
        pareto_df[metric_name] = pf_Y

        return pareto_df

    def plot_combined_pareto_fronts(df_fairgrid, metrics_to_include, visual_config):
        """
        Plot a combined Pareto front and individual plots for all fairness metrics selected by the user.
        
        Parameters:
        - df_fairgrid: DataFrame from FairGridCV
        - metrics_to_include: Dictionary with keys as fairness metric names and values as boolean indicating if the user selected the metric or not.
        - visual_config: Dictionary with visual properties (color, size, symbol) and the respective hyperparameters they map to.
        
        Returns:
        - List of Plotly figures (with the last figure being the combined plot)
        """
        
        # Filter out the metrics selected by the user
        selected_metrics = [metric for metric, selected in metrics_to_include.items() if selected]
        
        figures = []
        
        # Create an empty Plotly figure for combined plot
        fig = go.Figure()
        combined_fig = go.Figure()
        
        # List of colors (extend this if you have more metrics)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
        
        for idx, metric in enumerate(selected_metrics):
            # Create an empty figure for each metric
            fig = go.Figure()

            # Generate Pareto front for the current fairness metric
            pareto_df = pareto_frontier(df_fairgrid, metric)

            # Add Pareto frontier to the figure
            fig.add_trace(go.Scatter(x=pareto_df["Accuracy"], 
                                    y=pareto_df[metric],
                                    mode='lines+markers',
                                    marker=dict(size=11, line=dict(width=2, color='DarkSlateGray'), opacity=1),
                                    name=f"Pareto {metric}",
                                    line=dict(color='DarkSlateGray', width=2)))

            # Individual plot for current metric using visual configuration
            all_comb = px.scatter(df_fairgrid, 
                                x="Accuracy", 
                                y=metric, 
                                color=visual_config.get('color', None), 
                                symbol=visual_config.get('symbol', None),
                                hover_data=visual_config.get('hover_data', []))
        
            # Add all combination traces to the figure
            for trace in all_comb.data:
                # Extract hyperparameter values that determine color and symbol
                color_hp_value = trace.customdata[0][visual_config.get('hover_data', []).index(visual_config['color'])] if 'color' in visual_config else None
                symbol_hp_value = trace.customdata[0][visual_config.get('hover_data', []).index(visual_config['symbol'])] if 'symbol' in visual_config else None
                
                # Construct the hyperparameter description string
                descriptions = []
                # Comment out or remove the following lines to prevent the color hyperparameter from showing in the legend
                # if color_hp_value is not None:
                #     descriptions.append(str(color_hp_value))
                if symbol_hp_value is not None:
                    descriptions.append(str(symbol_hp_value))
                hyperparameters_str = " + ".join(descriptions)
                trace.name = hyperparameters_str  # set the trace name to this string
            
                # Add the trace to the figure
                fig.add_trace(trace)

            # Adjust marker aesthetics (if needed)
            fig.update_traces(marker=dict(size=11,
                                line=dict(width=1.5,
                                color='DarkSlateGrey'), opacity=0.75),
                                selector=dict(mode='markers'))
            
            legend_title_text = "Hyperparameter"
            if 'symbol' in visual_config:
                legend_title_text = legend_title_text + ": " + visual_config['symbol']
            
            fig.update_layout(title='Single Model Pareto Front: ' + metric,
                              xaxis_title='Accuracy',
                              yaxis_title=metric,
                              legend_title=legend_title_text,  # updated legend title
                              coloraxis_colorbar=dict(x=-0.4))

            # Append the figure to the figures list
            figures.append(fig)
            
            # Add to the combined plot
            combined_fig.add_trace(go.Scatter(x=pareto_df["Accuracy"], 
                                            y=pareto_df[metric],
                                            mode='lines+markers',
                                            marker=dict(size=11,line=dict(width=1.5,color='DarkSlateGray'), opacity=1),
                                            name=metric,
                                            line=dict(color=colors[idx])))
        
        combined_fig.update_layout(title='Overlapped Pareto Fronts',
                                xaxis_title='Accuracy',
                                yaxis_title='Fairness Metric Value',
                                legend_title='Fairness Metrics')
        
        figures.append(combined_fig)  # Append the combined figure to the list

        return figures
    
    def plot_best_pareto_fronts(metrics_to_include, visual_configs, checkDT, checkSVC, checkLR, checkRF):

        selected_metrics = [metric for metric, selected in metrics_to_include.items() if selected]
        figures = []
        
        # Color map for models
        model_colors = {
            "Decision Tree": "red",
            "SVC": "blue",
            "Logistic Regression": "green",
            "Random Forest": "yellow"
        }
        
        # Base hover data
        base_hover_data = ["model"]
        
        # Define the hover data for each model
        model_hover_data = {}
        if checkDT:
            model_hover_data["Decision Tree"] = ["criterion", "max_depth"]
        if checkSVC:
            model_hover_data["SVC"] = ["kernel", "shrinking"]
        if checkLR:
            model_hover_data["Logistic Regression"] = ["C", "penalty"]
        if checkRF:
            model_hover_data["Random Forest"] = ["bootstrap", "max_depth"]

        for metric in selected_metrics:
            # Calculate the overall Pareto front using combined data
            pareto_df = pareto_frontier(combined_df, metric)
            
            fig = go.Figure()  # Create an empty figure
            
            # Add Pareto front with only the model name as hover data
            pareto_trace = go.Scatter(
                x=pareto_df["Accuracy"],
                y=pareto_df[metric],
                mode="markers",
                marker=dict(
                    size=11,
                    opacity=1,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                hoverinfo="x+y",
                name="Pareto Front"
            )
            fig.add_trace(pareto_trace)
            
            pareto_line = px.line(pareto_df, x="Accuracy", y=metric).data[0]
            fig.add_trace(pareto_line)
            fig.update_traces(line=dict(width=2, color='DarkSlateGray'))

            
            # Plot each model with its custom hovertemplate
            for model_name, hyperparams in model_hover_data.items():
                model_df = combined_df[combined_df["model"] == model_name]
                
                hovertemplate_parts = [f"<b>{model_name}</b>"]  # Start with the model name
                hovertemplate_parts.extend([f"{param}: %{{customdata[{i}]}}" for i, param in enumerate(hyperparams)])
                hovertemplate = "<br>".join(hovertemplate_parts)
                
                fig.add_trace(
                    go.Scatter(
                        x=model_df["Accuracy"],
                        y=model_df[metric],
                        customdata=model_df[hyperparams].values,
                        mode="markers",
                        marker=dict(size=11, opacity=0.75, line=dict(width=1.5, color='DarkSlateGrey')),
                        name=model_name,
                        marker_color=model_colors[model_name],
                        hovertemplate=hovertemplate
                    )
                )

            fig.update_layout(title='Combined Pareto front: ' + metric,
                                xaxis_title='Accuracy',
                                yaxis_title=metric,
                                legend_title='Fairness Metrics')
            
            figures.append(fig)
        
        return figures
    
    def plot_model_pareto_fronts(metrics_to_include, checkDT, checkSVC, checkLR, checkRF):
        selected_metrics = [metric for metric, selected in metrics_to_include.items() if selected]
        figures = []

        # Color map for models
        model_colors = {
            "Decision Tree": "red",
            "SVC": "blue",
            "Logistic Regression": "green",
            "Random Forest": "yellow"
        }
        
        for metric in selected_metrics:
            fig = go.Figure()  # Create an empty figure
            
            # Plot the Pareto front for each model
            for model_name, color in model_colors.items():
                if model_name == "Decision Tree" and not checkDT:
                    continue
                if model_name == "SVC" and not checkSVC:
                    continue
                if model_name == "Logistic Regression" and not checkLR:
                    continue
                if model_name == "Random Forest" and not checkRF:
                    continue

                model_df = combined_df[combined_df["model"] == model_name]
                pareto_df = pareto_frontier(model_df, metric)  # Get the Pareto front for the current model

                fig.add_trace(
                    go.Scatter(
                        x=pareto_df["Accuracy"],
                        y=pareto_df[metric],
                        mode="lines+markers",
                        marker=dict(size=11, opacity=1, color=color, line=dict(width=1.5, color='DarkSlateGrey')),
                        name=f"Pareto Front {model_name}"
                    )
                )

                fig.update_layout(title='Overlapped multi-model Pareto fronts: ' + metric,
                                xaxis_title='Accuracy',
                                yaxis_title=metric,
                                legend_title='Models')

            figures.append(fig)
        
        return figures

    if opt_method=='Grid Search':
        # Initialize an empty dictionary for classifiers
        classifiers_config = {}

        # Add configurations conditionally

        # Decision Tree
        if checkDT:  # Assuming `checkDT` is the condition for adding Decision Tree configurations
            classifiers_config["Decision Tree"] = {
                'classifier': DecisionTreeClassifier,
                'config_space': cs_dtc,
                'visual_config': {
                    'color': 'max_depth',
                    'symbol': 'max_features',
                    'hover_data': ['criterion', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_depth']
                },
                'filename': "df_dtc.csv",
            }

        # SVC
        if checkSVC:  # Assuming `checkSVC` is the condition for adding SVC configurations
            classifiers_config["SVC"] = {
                'classifier': SVC,
                'config_space': cs_svc,
                'visual_config': {
                    'color': 'C',
                    'symbol': 'kernel',
                    'hover_data': ['kernel', 'C', 'shrinking', 'degree', 'coef0', 'gamma']
                },
                'filename': "df_svc.csv",
            }

        # Logistic Regression
        if checkLR:  # Assuming `checkLR` is the condition for adding Logistic Regression configurations
            classifiers_config["Logistic Regression"] = {
                'classifier': LogisticRegression,
                'config_space': cs_lr,
                'visual_config': {
                    'color': 'C',
                    'symbol': 'fit_intercept',
                    'hover_data': ['penalty', 'fit_intercept', 'C']
                },
                'filename': "df_lr.csv",
            }

        # Random Forest
        if checkRF:  # Assuming `checkRF` is the condition for adding Random Forest configurations
            classifiers_config["Random Forest"] = {
                'classifier': RandomForestClassifier,
                'config_space': cs_rf,
                'visual_config': {
                    'color': 'max_depth',
                    'symbol': 'bootstrap',
                    'hover_data': ['bootstrap', 'max_samples', 'criterion', 'max_depth', 'max_features', 'min_samples_split', 'min_samples_leaf']
                },
                'filename': "df_rfc.csv",
            }


        model_to_df_name = {
            "Decision Tree": "df_dtc",
            "SVC": "df_svc",
            "Logistic Regression": "df_lr",
            "Random Forest": "df_rfc"
        }

        # Loop through the classifiers and their configurations
        # Loop through the classifiers and their configurations
        for model_name, config in classifiers_config.items():
            if opt_method == 'Grid Search':
                check = {
                    "Decision Tree": checkDT,
                    "SVC": checkSVC,
                    "Logistic Regression": checkLR,
                    "Random Forest": checkRF
                }

                if check[model_name]:
                    if model_name == "Decision Tree":
                        df_dtc = FairGridCV(X_scaled, y, X_scaled[X_sen], config['classifier'], n_seeds, config['config_space'], metrics_to_include)
                        fig = plotly_parallel_coordinates(df_dtc)
                        st.plotly_chart(fig)
                    elif model_name == "SVC":
                        df_svc = FairGridCV(X_scaled, y, X_scaled[X_sen], config['classifier'], n_seeds, config['config_space'], metrics_to_include)
                        pfig = plotly_parallel_coordinates(df_svc)
                        st.plotly_chart(fig)
                    elif model_name == "Logistic Regression":
                        df_lr = FairGridCV(X_scaled, y, X_scaled[X_sen], config['classifier'], n_seeds, config['config_space'], metrics_to_include)
                        fig = plotly_parallel_coordinates(df_lr)
                        st.plotly_chart(fig)
                    elif model_name == "Random Forest":
                        df_rfc = FairGridCV(X_scaled, y, X_scaled[X_sen], config['classifier'], n_seeds, config['config_space'], metrics_to_include)
                        fig = plotly_parallel_coordinates(df_rfc)
                        st.plotly_chart(fig)
                    
                    current_df = locals()[model_to_df_name[model_name]]
                    st.write('---')
                    st.subheader(model_name)
                    st.write('---')
                    st.write('DataFrame containing the tested points.')
                    st.write(current_df.head(10))
                    csv = convert_df(current_df)
                    download_button_str = download_button(csv, config['filename'], 'Download Table', pickle_it=False)
                    st.markdown(download_button_str, unsafe_allow_html=True)
                    st.write('---')

                    current_df["model"] = model_name

                    figures = plot_combined_pareto_fronts(current_df, metrics_to_include, config['visual_config'])

                    for fig in figures:
                        st.plotly_chart(fig)

        all_dataframes = []

        # Directly append the DataFrames if they've been created
        if 'df_dtc' in locals():
            all_dataframes.append(df_dtc)
        if 'df_svc' in locals():
            all_dataframes.append(df_svc)
        if 'df_lr' in locals():
            all_dataframes.append(df_lr)
        if 'df_rfc' in locals():
            all_dataframes.append(df_rfc)

        # Combine dataframes if they exist
        if all_dataframes:
            if len(all_dataframes) > 1:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
            else:
                combined_df = all_dataframes[0]  # If only one dataframe exists, just select that
        else:
            # Handle the case when none of the dataframes exists
            combined_df = pd.DataFrame()  # Create an empty dataframe

        visual_configs = {
        'hover_data': ['model', 'criterion', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_depth', 'kernel', 'C', 'shrinking', 'degree', 'coef0', 'gamma', 'penalty', 'bootstrap', 'max_samples'] 
        }

        st.write('---')

        st.subheader('Combined Pareto Front')

        st.write('---')

        figures = plot_best_pareto_fronts(metrics_to_include, visual_configs, checkDT, checkSVC, checkLR, checkRF)

        for fig in figures:
            st.plotly_chart(fig)

        st.write('---')

        st.subheader('Overlapped Multi-Model Pareto Fronts')

        st.write('---')

        figures = plot_model_pareto_fronts(metrics_to_include, checkDT, checkSVC, checkLR, checkRF)

        # Display each plot in Streamlit
        for fig in figures:
            st.plotly_chart(fig)

        st.write('---')

###############################################
###############################################
###--------- BAYESIAN OPTIMIZATION ---------###
###############################################
###############################################

    if opt_method == 'MOBO':
        
        def plot_individual_pareto_fronts_MOBO(df_fairgrid, fairness_metric, visual_config):
            """
            Plot an individual Pareto front for a specific fairness metric.
            
            Parameters:
            - df_fairgrid: DataFrame containing the Pareto points
            - fairness_metric: The fairness metric for which the Pareto front is to be plotted
            - visual_config: Dictionary with visual properties (color, size, symbol) and the respective hyperparameters they map to.
            
            Returns:
            - Plotly figure for the individual Pareto front
            """

            df_fairgrid = df_fairgrid.sort_values(by='Accuracy', ascending=True)
            # Create an empty Plotly figure
            fig = go.Figure()

            # Add Pareto points to the figure
            fig.add_trace(go.Scatter(x=df_fairgrid["Accuracy"], 
                                    y=df_fairgrid[fairness_metric],
                                    mode='lines+markers',
                                    marker=dict(size=11, line=dict(width=2, color='DarkSlateGray'), opacity=1),
                                    name=f"Pareto {fairness_metric}",
                                    line=dict(color='DarkSlateGray', width=2)))

            # Individual plot for current metric using visual configuration
            all_comb = px.scatter(df_fairgrid, 
                                x="Accuracy", 
                                y=fairness_metric, 
                                color=visual_config.get('color', None), 
                                symbol=visual_config.get('symbol', None),
                                hover_data=visual_config.get('hover_data', []))

            # Add all combination traces to the figure
            for trace in all_comb.data:
                # Extract hyperparameter values that determine color and symbol
                color_hp_value = trace.customdata[0][visual_config.get('hover_data', []).index(visual_config['color'])] if 'color' in visual_config else None
                symbol_hp_value = trace.customdata[0][visual_config.get('hover_data', []).index(visual_config['symbol'])] if 'symbol' in visual_config else None
                
                # Construct the hyperparameter description string
                descriptions = []
                if color_hp_value is not None:
                    descriptions.append(str(color_hp_value))
                if symbol_hp_value is not None:
                    descriptions.append(str(symbol_hp_value))
                hyperparameters_str = " + ".join(descriptions)
                trace.name = hyperparameters_str  # set the trace name to this string

                # Add the trace to the figure
                fig.add_trace(trace)

            # Adjust marker aesthetics (if needed)
            fig.update_traces(marker=dict(size=11,
                                        line=dict(width=1.5,
                                        color='DarkSlateGrey'), opacity=0.75),
                            selector=dict(mode='markers'))
            
            fig.update_layout(title='Individual Pareto Front: ' + fairness_metric,
                                xaxis_title='Accuracy',
                                yaxis_title=fairness_metric,
                                legend_title='Hyperparameters',
                                coloraxis_colorbar=dict(x=-0.4))
            return fig


        def plot_overlapped_pareto_front_MOBO(all_dfs):
            """
            Plot the overlapped Pareto front for all fairness metrics.
            
            Parameters:
            - all_dfs: Dictionary with fairness metrics as keys and their respective DataFrames as values.
            - visual_config: Dictionary with visual properties (color, size, symbol) and the respective hyperparameters they map to.
            
            Returns:
            - Plotly figure for the overlapped Pareto front
            """

            combined_fig = go.Figure()
            
            # List of colors (extend this if you have more metrics)
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
            
            for idx, (fairness_metric, df_fairgrid) in enumerate(all_dfs.items()):
                df_fairgrid = df_fairgrid.sort_values(by='Accuracy', ascending=True)
                # Since df_fairgrid is already the Pareto front, you can directly use it for plotting
                combined_fig.add_trace(go.Scatter(x=df_fairgrid["Accuracy"], 
                                                y=df_fairgrid[fairness_metric],
                                                mode='lines+markers',
                                                marker=dict(size=11, line=dict(width=1.5, color='DarkSlateGray'), opacity=1),
                                                name=fairness_metric,
                                                line=dict(color=colors[idx])))

            combined_fig.update_layout(title='Overlapped Pareto Fronts',
                                    xaxis_title='Accuracy',
                                    yaxis_title='Fairness Metric Value',
                                    legend_title='Fairness Metrics')
            
            return combined_fig
        
        def plot_overlapped_pareto_by_model(model_data, fairness_metric):
            combined_fig = go.Figure()
            model_colors = {
                "DecisionTreeClassifier": "red",
                "SVC": "blue",
                "LogisticRegression": "green",
                "RandomForestClassifier": "yellow"
            }
            
            all_points = []  # to store all points from all models
            
            for model_name, data in model_data.items():
                if fairness_metric in data:
                    # Sort the data points by accuracy
                    sorted_indices = sorted(range(len(data[fairness_metric]["x"])), key=lambda k: data[fairness_metric]["x"][k])
                    sorted_x = [data[fairness_metric]["x"][i] for i in sorted_indices]
                    sorted_y = [data[fairness_metric]["y"][i] for i in sorted_indices]
                    
                    # Add points to all_points list
                    all_points.extend(list(zip(sorted_x, sorted_y)))
                    
                    combined_fig.add_trace(go.Scatter(
                        x=sorted_x, 
                        y=sorted_y,
                        mode='lines+markers',
                        marker=dict(size=11, line=dict(width=1.5, color='DarkSlateGray'), opacity=1),
                        name=model_name,
                        line=dict(color=model_colors.get(model_name, "black"))
                    ))

            # Compute the overall Pareto front from all_points
            pareto_points = compute_pareto_front(all_points)
            pareto_x, pareto_y = zip(*pareto_points)
            
            combined_fig.add_trace(go.Scatter(
                x=pareto_x, 
                y=pareto_y,
                mode='lines',
                name='Overall Best Pareto Front',
                line=dict(color='black', dash='dash')
            ))

            combined_fig.update_layout(
                title=f'Overlapped Pareto Fronts for {fairness_metric}',
                xaxis_title='Accuracy',
                yaxis_title=fairness_metric,
                legend_title='Models'
            )
            
            return combined_fig

        def compute_pareto_front(points):
                """
                Compute the Pareto front for a set of points.
                
                Parameters:
                - points: List of tuples, where each tuple is (accuracy, fairness_metric_value)
                
                Returns:
                - Pareto front as a list of tuples
                """
                pareto_front = []
                points_sorted = sorted(points, key=lambda x: (-x[0], -x[1]))  # Sort by accuracy descending and then by fairness_metric_value descending
                
                for point in points_sorted:
                    if not pareto_front:
                        pareto_front.append(point)
                    else:
                        if point[1] > pareto_front[-1][1]:  # Check if the fairness metric value is higher than the last point in the Pareto front
                            pareto_front.append(point)
                            
                return pareto_front

        def decisiontree_config():
            space = sp.Space()
            criterion = sp.Categorical("criterion", ["gini", "entropy", "log_loss"], default_value="gini")
            max_depth = sp.Int("max_depth", 1, 20, default_value=6)
            splitter = sp.Categorical("splitter", ["random", "best"], default_value="best")
            max_features = sp.Int("max_features", 1, 20, default_value=20)
            min_samples_split = sp.Int("min_samples_split", 2, 40, default_value=20)
            # min_samples_leaf = sp.Int("min_samples_leaf", 1, 40, default_value=20)
            space.add_variables([criterion, max_depth, splitter, max_features, min_samples_split])
            return space
        
        def randomforest_config():
            space = sp.Space()
            criterion = sp.Categorical("criterion", ["gini", "entropy"], default_value="gini")
            max_depth = sp.Int("max_depth", 1, 20, default_value=6)
            #  bootstrap = sp.Categorical("bootstrap", [True, False], default_value=True)
            max_samples = sp.Real("max_samples", 0.5, 1.0, default_value=1.0)
            max_features = sp.Int("max_features", 1, 20, default_value=20)
            min_samples_split = sp.Int("min_samples_split", 2, 40, default_value=20)
            # min_samples_leaf = sp.Int("min_samples_leaf", 1, 40, default_value=20)

            #cond_max_samples = EqualsCondition(max_samples, bootstrap, True)  # x2 is active when x1 = c1

            space.add_variables([criterion, max_depth, max_samples, max_features, min_samples_split])
            #space.add_conditions([cond_max_samples])
            
            return space

        def logisticregression_config():
            space = sp.Space()
            # solver = sp.Categorical("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], default_value="newton-cg")
            C = sp.Real("C", 0.01, 500.0, default_value=1.0, log=True)
            penalty = sp.Categorical("penalty", ['elasticnet'], default_value='elasticnet')
            fit_intercept = sp.Categorical("fit_intercept", [True, False], default_value=True)
            l1_ratio = sp.Real("l1_ratio", 0.0, 1.0, default_value=0.0)

            space.add_variables([C, fit_intercept, l1_ratio])
                
            return space

        def SVC_config():
            space = sp.Space()
            C = sp.Real("C", 0.001, 1000, default_value=1.0, log=True)
            #   kernel = sp.Categorical("kernel", ['rbf', 'poly', 'sigmoid'], default_value='rbf')
            gamma = sp.Real("gamma", 0.001, 1000, default_value=1.0, log=True)
            
            space.add_hyperparameters([C, gamma])
            
            return space
        
        # List of fairness metrics
        fairness_metrics_list = [
            'Predictive parity',
            'Predictive equality',
            'Equal opportunity',
            'Accuracy equality',
            'Equalized odds',
            'Statistical parity'
        ]

        # List of selected classifiers based on checkboxes
        selected_classifiers = []
        if checkRF:
            selected_classifiers.append('Random Forest Classifier')
        if checkLR:
            selected_classifiers.append('Logistic Regression')
        if checkSVC:
            selected_classifiers.append('Support Vector Classifier')
        if checkDT:
            selected_classifiers.append('Decision Tree Classifier')


        # List of classifiers and their configurations
        classifiers = {
            'DecisionTreeClassifier': {
                'config': decisiontree_config,
                'classifier': DecisionTreeClassifier,
                'params': ['criterion', 'max_depth', 'splitter', 'max_features', 'min_samples_split']
            },
            'RandomForestClassifier': {
                'config': randomforest_config,
                'classifier': RandomForestClassifier,
                'params': ['criterion', 'max_depth', 'max_samples', 'max_features', 'min_samples_split']
            },
            'LogisticRegression': {
                'config': logisticregression_config,
                'classifier': LogisticRegression,
                'params': ['C', 'fit_intercept', 'l1_ratio']
            },
            'SVC': {
                'config': SVC_config,
                'classifier': SVC,
                'params': ['C', 'gamma']
            }
        }

        results = {}  # Dictionary to store results
        pareto_fronts = {}
        model_data = {}

        # Outer loop for each classifier
        for clf_name, clf_info in classifiers.items():
            # Check if the classifier is selected
            if (clf_name == 'DecisionTreeClassifier' and checkDT) or \
            (clf_name == 'RandomForestClassifier' and checkRF) or \
            (clf_name == 'LogisticRegression' and checkLR) or \
            (clf_name == 'SVC' and checkSVC):
                
                all_dfs = {}

                if clf_name == 'DecisionTreeClassifier':
                    st.write('---')
                    st.subheader('Decision Tree Classifier')
                elif clf_name == 'RandomForestClassifier':
                    st.write('---')
                    st.subheader('Random Forest Classifier')
                elif clf_name == 'LogisticRegression':
                    st.write('---')
                    st.subheader('Logistic Regression')
                else:
                    st.write('---')
                    st.subheader('Support Vector Classifier')

                # Inner loop for each fairness metric
                for fairness_metric in fairness_metrics_list:
                    if metrics_to_include[fairness_metric]:
                        def objective_function(config: sp.Configuration, clf_info, n_folds=n_seeds, sensitive_attribute=X_sen, features=X_scaled, target=y):

                            classifier_class = clf_info['classifier']
                            param_names = clf_info['params']

                            # Extract the parameters from the configuration
                            params = {param_name: config[param_name] for param_name in param_names}
                            params['random_state'] = 0

                            result = dict()

                            test_scores = []

                            # Initialize fairness aggregates
                            fairness_aggregates = {
                                'Predictive parity': [],
                                'Predictive equality': [],
                                'Equal opportunity': [],
                                'Accuracy equality': [],
                                'Equalized odds': [],
                                'Statistical parity': []
                            }

                            # Get metrics to include from Streamlit sidebar
                            metrics_to_include = {
                                'Predictive parity': metrics_checkboxes['Predictive Parity'],
                                'Predictive equality': metrics_checkboxes['Predictive Equality'],
                                'Equal opportunity': metrics_checkboxes['Equal Opportunity'],
                                'Accuracy equality': metrics_checkboxes['Accuracy Equality'],
                                'Equalized odds': metrics_checkboxes['Equalized Odds'],
                                'Statistical parity': metrics_checkboxes['Statistical Parity']
                            }

                            for train_index, test_index in StratifiedKFold(n_splits=n_folds).split(features, target):
                                X_train, X_test = features.loc[train_index], features.loc[test_index]
                                y_train, y_test = target.loc[train_index], target.loc[test_index]

                                # Locate privileged and unprivileged groups
                                X1_test = X_test.loc[(features[sensitive_attribute] > 0)]
                                X0_test = X_test.loc[(features[sensitive_attribute] == 0)]
                                y1_test = y_test.loc[(features[sensitive_attribute] > 0)]
                                y0_test = y_test.loc[(features[sensitive_attribute] == 0)]

                                # Fit the classifier
                                clf = classifier_class(**params)
                                clf.fit(X_train, y_train)

                                test_scores.append(clf.score(X_test, y_test))

                                # Predict the labels for the test set
                                y1_pred = clf.predict(X1_test)
                                y0_pred = clf.predict(X0_test)

                                # Compute fairness metrics
                                fairness_metrics = fairness(y1_pred, y1_test, y0_pred, y0_test, metrics_to_include)

                                # Accumulate the fairness metrics over each fold
                                for metric, value in fairness_metrics.items():
                                    fairness_aggregates[metric].append(value)

                            accuracy = sum(test_scores) / len(test_scores)

                            # Aggregate the fairness metrics over all folds
                            aggregated_fairness = {metric: np.mean(values) for metric, values in fairness_aggregates.items()}

                            # Only consider the current fairness metric and accuracy as objectives
                            result['objectives'] = [1 - accuracy, aggregated_fairness[fairness_metric]]

                            return result
                        
                        config_function = clf_info['config']

                        opt = Optimizer(
                            lambda config: objective_function(config, clf_info),
                            config_function(),
                            num_objectives=2,
                            num_constraints=0,
                            max_runs=100,
                            advisor_type='default',
                            surrogate_type='prf',
                            acq_type='parego',
                            acq_optimizer_type='local_random',
                            initial_runs=10,
                            init_strategy='random_explore_first',
                            task_id='moc',
                            random_state=42,
                            ref_point=[1,1],
                            )
                    
                        history = opt.run()

                        data = []
                        observations = history.get_pareto()
                        for obs in observations:
                            # Extract the configuration values
                            config_values = obs.config.get_dictionary()
                            
                            # Extract objectives and other attributes
                            obj1, obj2 = obs.objectives
                            
                            # Create a dictionary with all the extracted values
                            row = {'Accuracy': 1-obj1, fairness_metric: 1-obj2}
                            for param in clf_info['params']:
                                row[param] = config_values[param]
                            data.append(row)

                                # Adjust the visual configuration based on the classifier
                        if clf_name == 'DecisionTreeClassifier':
                            visual_config = {
                                'color': 'criterion',
                                'symbol': 'splitter',
                                'hover_data': ['criterion', 'max_depth', 'splitter']
                            }
                        elif clf_name == 'RandomForestClassifier':
                            visual_config = {
                                'color': 'max_depth',
                                'symbol': 'criterion',
                                'hover_data': ['criterion', 'max_depth', 'max_samples']
                            }
                        elif clf_name == 'LogisticRegression':
                            visual_config = {
                                'color': 'C',
                                'symbol': 'fit_intercept',
                                'hover_data': ['C', 'l1_ratio', 'fit_intercept']
                            }
                        elif clf_name == 'SVC':
                            visual_config = {
                                'color': 'C',
                                'symbol': 'gamma',
                                'hover_data': ['C', 'gamma']
                            }

                        # Convert the list of dictionaries to a DataFrame
                        st.write('---')
                        st.write('DataFrame containing the best points.')
                        df_fairgrid = pd.DataFrame(data)
                        pareto_fronts[clf_name] = df_fairgrid

                        st.write(df_fairgrid)
                        csv = convert_df(df_fairgrid)
                        filename = f"df_{clf_name.lower().replace('classifier', 'clf')}.csv"
                        download_button_str = download_button(csv, filename, 'Download Table', pickle_it=False)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                        st.write('---')

                        # Step 2: Use plot_individual_pareto_fronts_MOBO inside the fairness metric loop
                        individual_fig = plot_individual_pareto_fronts_MOBO(df_fairgrid, fairness_metric, visual_config)
                        st.plotly_chart(individual_fig)

                        # Inside the fairness metric loop, after plotting the individual figure
                        if clf_name not in model_data:
                            model_data[clf_name] = {}
                        model_data[clf_name][fairness_metric] = {
                            "x": df_fairgrid["Accuracy"].tolist(),
                            "y": df_fairgrid[fairness_metric].tolist()
                        }

                        # Store the dataframe for later use
                        all_dfs[fairness_metric] = df_fairgrid

                overlapped_fig = plot_overlapped_pareto_front_MOBO(all_dfs)
                st.plotly_chart(overlapped_fig)
            all_dfs.clear()     

        selected_metrics = [metric for metric, selected in metrics_to_include.items() if selected]

        st.write('---')

        st.subheader('Mutli-Model Pareto Fronts')

        st.write('---')

        for fairness_metric in selected_metrics:
            fig = plot_overlapped_pareto_by_model(model_data, fairness_metric)
            st.plotly_chart(fig)

        st.write('---')

        st.balloons()
