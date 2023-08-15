import streamlit as st
from numpy import np

st.sidebar.header('FairPilot Settings')

st.sidebar.subheader('Choose Learning Methods')
checkLR = st.sidebar.checkbox('Logistic Regression')
if checkLR:
    with st.sidebar.expander('**Set HP space:**'):
        st.multiselect('Penalty', ['None', 'l2'], ['None'])
        st.multiselect('C', ['1000', '100', '10', '1', 
                             '0.1', '0.01', '0.001'], ['1'])

checkDT = st.sidebar.checkbox('Decision Tree Classifier')
if checkDT:
    with st.sidebar.expander('**Set HP space:**'):
        criterion = st.multiselect('Criterion', ['gini', 'entropy'])
        class_weight = list(st.multiselect('Class-Weight', [None, 'balanced']))
        max_features = np.array(st.multiselect('Max-features', ['log2', 'sqrt']))
        min_samples_split = np.array(st.slider('Min-samples split', 2, 20, [2, 10], 1))
        min_samples_split = range(min_samples_split[0], min_samples_split[1])
        min_samples_leaf = np.array(st.slider('Min-samples leaf', 1, 20, [5, 10], 1))
        min_samples_leaf = range(min_samples_leaf[0], min_samples_leaf[1])

checkRF = st.sidebar.checkbox('Random Forest Classifier')
if checkRF:
    with st.sidebar.expander('**Set HP space:**'):
        st.multiselect('Bootstrap', ['True', 'False'])
        st.multiselect('Criterion ', ['gini', 'entropy'])
        st.multiselect('Class-Weight ', ['None', 'balanced'])
        st.multiselect('Max-features ', ['None', 'log2', 'sqrt'])
        st.slider('Min-samples split ', 2, 20, [5, 10], 1)
        st.slider('Min-samples leaf ', 1, 20, [5, 10], 1)

checkSVC = st.sidebar.checkbox('Support Vector Classifier')
if checkSVC:
    with st.sidebar.expander('**Set HP space:**'):
        st.multiselect('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        st.multiselect('C ', ['1000', '100', '10', '1', 
                            '0.1', '0.01', '0.001'], ['1'])

checkNN = st.sidebar.checkbox('Neural Network')
if checkNN:
    with st.sidebar.expander('**Set HP space:**'):
        st.multiselect('Layer 1 nodes', ['64', '128', '256'], ['64'])
        st.multiselect('Layer 2 nodes', ['64', '128', '256'], ['64'])
        st.slider('Batch size', 4, 512, [16, 128], 2)
        st.slider('Layer 1 dropout', 0.1, 0.5, [0.2, 0.4], 0.1)
        st.slider('Layer 2 dropout', 0.1, 0.5, [0.2, 0.4], 0.1)
        st.slider('Epochs', 5, 1000, [10, 100], 5)

st.sidebar.write('---')

st.sidebar.subheader('Choose Fairness Metrics')
st.sidebar.checkbox('Predictive Parity',
                    help='Probability of a positive outcome should be the same for both classes.')
st.sidebar.checkbox('Predictive Equality', 
                    help='The fraction of false positive (or true negative) predictions should be the same for both classes.')
st.sidebar.checkbox('Equal Opportunity',
                    help='The fraction of false negative (or true positive) predictions should be the same for both classes.')
st.sidebar.checkbox('Accuracy Equality',
                    help='Prediction accuracy should be the same for both classes.')
st.sidebar.checkbox('Equalized Odds',
                    help='The fraction of false positive AND true positive predictions should be the same for both classes.')
st.sidebar.checkbox('Statistical Parity',
                    help='The ratio of positive predictions should be the same regardless of the class.')

st.sidebar.write('---')

st.sidebar.subheader('Advanced Settings')
opt_methods =st.sidebar.multiselect('**Optimization Algorithm**',
                                    ['Grid Search', 'Random Search', 'BO', 'Fair-BO'],['Grid Search'])
split_size = st.sidebar.slider('**Data Split Ratio (% of Training Set)**', 50, 90, 80, 5)
n_seeds = st.sidebar.slider('**Number of Repetitions**', 2, 20, 5, 1)
scl_type = st.sidebar.selectbox('**Data Scaling**',('None', 'Standardization', 'Normalization'))
imp_type = st.sidebar.selectbox('**Data Imputation**',('None', 'Median', 'Mean'))
colorblind = st.sidebar.selectbox('**Color-Blind Friendly Plots**',('Yes', 'No'))
graph_res = st.sidebar.selectbox('**Resolution of Exported Plots**',('High', 'Medium', 'Low', 'Max'))
report = st.sidebar.selectbox('**Generate a Report**',('Yes', 'No'))