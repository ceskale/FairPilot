import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def MMPFBO(df_cum, df_MODEL, name):

    fig_PT = go.Scatter(
    line_shape="vh", 
    #line_dash="dash",
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
            line_shape="vh", 
            line_dash="dash",
            mode='lines+markers',
            x=df_dt['accuracy'],
            y=df_dt[name],
            name='Decision Tree',
            marker=dict(size=6, color="red", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            line=dict(width=1, color='red'),
            customdata = np.stack((df_dt['min_samples_leaf'], df_dt['min_samples_split'], df_dt['max_features'], 
                                df_dt['criterion'], df_dt['max_depth'], df_dt['splitter']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'min_samples_leaf: %{customdata[0]} <br>' +
            'min_samples_split: %{customdata[1]} <br>' +
            'max_features: %{customdata[2]} <br>' +
            'criterion: %{customdata[3]} <br>' +
            'max_depth: %{customdata[4]} <br>' +
            'splitter: %{customdata[5]} <br>' +
            '<extra>ok</extra>'),
        )

    if df_MODEL[1].empty :
        fig_svc = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_svc = df_MODEL[1]
        fig_svc = go.Scatter(
            line_shape="vh", 
            line_dash="dash",
            mode='lines+markers',
            line=dict(width=1, color='hotpink'),
            x=df_svc['accuracy'],
            y=df_svc[name],
            name='Support Vector Classifier',
            marker=dict(size=6, color="hotpink", symbol='circle',line=dict(width=1,
                                        color='darkslategray')),
            customdata = np.stack((df_svc['penalty'], df_svc['C'], df_svc['loss'], df_svc['fit_intercept'], df_svc['intercept_scaling']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'penalty: %{customdata[0]} <br>' +
            'C: %{customdata[1]} <br>' +
            'loss: %{customdata[2]} <br>' +
            'fit_intercept: %{customdata[3]} <br>' +
            'intercept_scaling: %{customdata[4]} <br>' +
            '<extra>ok</extra>'),
        )

    if df_MODEL[2].empty :
        fig_lr = go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")
    else:
        df_lr = df_MODEL[2]
        fig_lr = go.Scatter(
            line_shape="vh", 
            line_dash="dash",
            mode='lines+markers',
            x=df_lr['accuracy'],
            y=df_lr[name],
            name='Logistic Regression',
            line=dict(width=1, color='green'),
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
            line_shape="vh", 
            line_dash="dash",
            mode='lines+markers',
            x=df_rf['accuracy'],
            y=df_rf[name],
            name='Random Forest',
            line=dict(width=1, color='gold'),
            marker=dict(size=6, color="gold", symbol='circle',line=dict(width=1.5,
                                        color='darkslategray')),
            customdata = np.stack((df_rf['min_samples_leaf'], df_rf['min_samples_split'], df_rf['max_features'], 
                                df_rf['criterion'], df_rf['max_depth'], df_rf['bootstrap'], df_rf['max_samples']), axis=-1),
            hovertemplate = ('Accuracy: %{x:,.4f} <br>' + 
            'Statistical parity: %{y:,.4f} <br>' + 
            'min_samples_leaf: %{customdata[0]} <br>' +
            'min_samples_split: %{customdata[1]} <br>' +
            'max_features: %{customdata[2]} <br>' +
            'criterion: %{customdata[3]} <br>' +
            'max_depth: %{customdata[4]} <br>' +
            'bootstrap: %{customdata[5]} <br>' +
            'max_samples: %{customdata[6]} <br>' +
            '<extra>ok</extra>'),
        )
    
    data_legend = [fig_PT, fig_dt, fig_svc, fig_lr, fig_rf]

    fig = go.Figure(data=data_legend)

    fig.update_layout(yaxis_visible=True)

    fig.update_yaxes(title=name, title_font=dict(size=20))
    fig.update_xaxes(title='Accuracy', title_font=dict(size=20))
    fig.update_layout(yaxis = dict(tickfont = dict(size=15)))
    fig.update_layout(xaxis = dict(tickfont = dict(size=15)))

    return fig