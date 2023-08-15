import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def MMPF(df_cum, df_MODEL, name):

    fig_PT = go.Scatter(
    line_shape="vh", 
    mode='lines+markers',
    x=df_cum['Accuracy'],
    y=df_cum[name],
    name='Pareto Front',
    marker=dict(size= 6, color='darkslategray',line=dict(width=2.5,
                                    color='darkslategray')),
    line=dict(width=2.5)
    )

    if df_MODEL[0].empty :
        fig_dt = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_dt = df_MODEL[0]
        fig_dt = go.Scatter(
            mode='markers',
            x=df_dt['Accuracy'],
            y=df_dt[name],
            name='Decision Tree',
            marker=dict(size=6, color="red", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            customdata = np.stack((df_dt['min_samples_leaf'], df_dt['min_samples_split'], df_dt['max_features'], 
                                df_dt['criterion'], df_dt['class_weight'], df_dt['max_depth']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'min_samples_leaf: %{customdata[0]} <br>' +
            'min_samples_split: %{customdata[1]} <br>' +
            'max_features: %{customdata[2]} <br>' +
            'criterion: %{customdata[3]} <br>' +
            'class_weight: %{customdata[4]} <br>' +
            '<extra>ok</extra>'),
        )

    if df_MODEL[1].empty :
        fig_svc = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_svc = df_MODEL[1]
        fig_svc = go.Scatter(
            mode='markers',
            x=df_svc['Accuracy'],
            y=df_svc[name],
            name='Support Vector Classifier',
            marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            customdata = np.stack((df_svc['penalty'], df_svc['C'], df_svc['loss'], df_svc['fit_intercept'], df_svc['intercept_scaling']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'kernel: %{customdata[0]} <br>' +
            'C: %{customdata[1]} <br>' +
            '<extra>ok</extra>'),
        )

    if df_MODEL[2].empty :
        fig_lr = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_lr = df_MODEL[2]
        fig_lr = go.Scatter(
            mode='markers',
            x=df_lr['Accuracy'],
            y=df_lr[name],
            name='Logistic Regression',
            marker=dict(size=6, color="green", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            customdata = np.stack((df_lr['penalty'], df_lr['C']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'penalty: %{customdata[0]} <br>' +
            'C: %{customdata[1]} <br>' +
            '<extra>ok</extra>'),
        )

    if df_MODEL[3].empty :
        fig_rf = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_rf = df_MODEL[3]
        fig_rf = go.Scatter(
            mode='markers',
            x=df_rf['Accuracy'],
            y=df_rf[name],
            name='Random Forest',
            marker=dict(size=6, color="gold", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            customdata = np.stack((df_rf['min_samples_leaf'], df_rf['min_samples_split'], df_rf['max_depth'], df_rf['max_features'], 
                                df_rf['criterion'], df_rf['class_weight'], df_rf['bootstrap'], df_rf['max_samples']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'min_samples_leaf: %{customdata[0]} <br>' +
            'min_samples_split: %{customdata[1]} <br>' +
            'max_depth: %{customdata[2]} <br>' +
            'max_features: %{customdata[3]} <br>' +
            'criterion: %{customdata[4]} <br>' +
            'class_weight: %{customdata[5]} <br>' +
            'bootstrap: %{customdata[6]} <br>' +
            'max_samples: %{customdata[7]} <br>' +
            '<extra>ok</extra>'),
        )

    if df_MODEL[4].empty :
        fig_NN = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_NN = df_MODEL[4]
        fig_NN = go.Scatter(
            mode='markers',
            x=df_NN['Accuracy'],
            y=df_NN[name],
            name='Neural Network',
            marker=dict(size=6, color="lime", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            customdata = np.stack((df_NN['L1_nodes'], df_NN['L2_nodes'], df_NN['L1_dropout_rate'], 
                                df_NN['L2_dropout_rate'], df_NN['batch_size'], df_NN['epochs']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            '{name}: %{y:,.4f} <br>' + 
            'L1_nodes: %{customdata[0]} <br>' +
            'L2_nodes: %{customdata[1]} <br>' +
            'L1_dropout_rate: %{customdata[2]} <br>' +
            'L2_dropout_rate: %{customdata[3]} <br>' +
            'batch_size: %{customdata[4]} <br>' +
            'epochs: %{customdata[5]} <br>' +
            '<extra>ok</extra>'),
        )
    
    data_legend = [fig_PT, fig_dt, fig_svc, fig_lr, fig_rf, fig_NN]

    fig = go.Figure(data=data_legend)

    fig.update_layout(yaxis_visible=True)

    fig.update_yaxes(title=name, title_font=dict(size=20))
    fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
    fig.update_layout(yaxis = dict(tickfont = dict(size=15)))
    fig.update_layout(xaxis = dict(tickfont = dict(size=15)))

    return fig
