import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def SingleModelParetoNN(dfPareto, fmPareto, fm1, sem_fm1, fm2, sem_fm2, fm3, sem_fm3, fm4, sem_fm4, fm5, sem_fm5, fm6, sem_fm6,
                        mts, ets,
                        L1_nodes, L2_nodes, L1_dropout_rate, L2_dropout_rate, batch_size, epochs, df_tot, name) :

    pareto_values_acc = np.unique(dfPareto['Accuracy'])
    pareto_ets = []

    pareto_prp = []
    pareto_prE = []
    pareto_Eop = []
    pareto_Acc = []
    pareto_Eq1 = []
    pareto_Pty = []

    pareto_sem_prp = []
    pareto_sem_prE = []
    pareto_sem_Eop = []
    pareto_sem_Acc = []
    pareto_sem_Eq1 = []
    pareto_sem_Pty = []

    pareto_L1_nodes = []
    pareto_L2_nodes = []
    pareto_L1_dropout_rate = []
    pareto_L2_dropout_rate = []
    pareto_batch_size = []
    pareto_epochs = []

    for i in range(len(pareto_values_acc)):
        index = np.where(pareto_values_acc[i] == mts)
        index2 = np.array(np.where(fmPareto == max(fmPareto[index])))
        pareto_ets.append(ets[index2[0,0]])
        pareto_prp.append(fm1[index2[0,0]])
        pareto_sem_prp.append(sem_fm1[index2[0,0]])
        pareto_prE.append(fm2[index2[0,0]])
        pareto_sem_prE.append(sem_fm2[index2[0,0]])
        pareto_Eop.append(fm3[index2[0,0]])
        pareto_sem_Eop.append(sem_fm3[index2[0,0]])
        pareto_Acc.append(fm4[index2[0,0]])
        pareto_sem_Acc.append(sem_fm4[index2[0,0]])
        pareto_Eq1.append(fm5[index2[0,0]])
        pareto_sem_Eq1.append(sem_fm5[index2[0,0]])
        pareto_Pty.append(fm6[index2[0,0]])
        pareto_sem_Pty.append(sem_fm6[index2[0,0]])
        pareto_L1_nodes.append(L1_nodes[index2[0,0]])
        pareto_L2_nodes.append(L2_nodes[index2[0,0]])
        pareto_L1_dropout_rate.append(L1_dropout_rate[index2[0,0]])
        pareto_L2_dropout_rate.append(L2_dropout_rate[index2[0,0]])
        pareto_batch_size.append(batch_size[index2[0,0]])
        pareto_epochs.append(epochs[index2[0,0]])
    

    solutions_space = {
        'Accuracy': pareto_values_acc, 
        'Accuracy SEM': pareto_ets,
        'Predictive parity': pareto_prp,
        'Predictive parity SEM': pareto_sem_prp,
        'Predictive equality': pareto_prE,
        'Predictive equality SEM': pareto_sem_prE,
        'Equal opportunity': pareto_Eop,
        'Equal opportunity SEM': pareto_sem_Eop,
        'Accuracy equality': pareto_Acc,
        'Accuracy equality SEM': pareto_sem_Acc,
        'Equalized odds': pareto_Eq1,
        'Equalized odds SEM': pareto_sem_Eq1,
        'Statistical parity': pareto_Pty,
        'Statistical parity SEM': pareto_sem_Pty,
        'L1_nodes': pareto_L1_nodes,
        'L2_nodes': pareto_L2_nodes,
        'L1_dropout_rate': pareto_L1_dropout_rate,
        'L2_dropout_rate': pareto_L2_dropout_rate,
        'batch_size': pareto_batch_size,
        'epochs': pareto_epochs
    }

    df = pd.DataFrame(data=solutions_space)

    fig1 = px.line(df, x="Accuracy", y=name, hover_data=['L1_nodes','L2_nodes','L1_dropout_rate','L2_dropout_rate','batch_size', 'epochs'], markers=True)
    fig1.update_traces(line_shape="vh")
    fig2 = px.scatter(df_tot, x = "Accuracy", y = name, color='epochs', symbol='batch_size',
                    hover_data=['L1_nodes','L2_nodes','L1_dropout_rate','L2_dropout_rate','batch_size', 'epochs'])
    fig3 = px.scatter(df_tot, x = "Accuracy", y = name, color='epochs', size='L1_dropout_rate', symbol='batch_size',
                    hover_data=['L1_nodes','L2_nodes','L1_dropout_rate','L2_dropout_rate','batch_size', 'epochs'])
    fig4 = px.scatter(df_tot, x = "Accuracy", y = name, color='batch_size', symbol='epochs',
                    hover_data=['L1_nodes','L2_nodes','L1_dropout_rate','L2_dropout_rate','batch_size', 'epochs'])
    fig5 = px.scatter(df_tot, x = "Accuracy", y = name, color='batch_size', size='L1_dropout_rate', symbol='epochs',
                    hover_data=['L1_nodes','L2_nodes','L1_dropout_rate','L2_dropout_rate','batch_size', 'epochs'])


    ntraces_a = df_tot['batch_size'].nunique()
    ntraces_b = df_tot['epochs'].nunique()

    fig = go.Figure()
    fig.add_trace(go.Scatter(fig1.data[0], visible=True))

    for i in range(ntraces_a):
        fig.add_trace(go.Scatter(fig2.data[i], visible=True))

    for i in range(ntraces_a):
        fig.add_trace(go.Scatter(fig3.data[i], visible=False))

    for i in range(ntraces_b):
        fig.add_trace(go.Scatter(fig4.data[i], visible=False))

    for i in range(ntraces_b):
        fig.add_trace(go.Scatter(fig5.data[i], visible=False))

    ntracesTOT = 1 + ntraces_a * 2 + ntraces_b * 2

    visible = []

    for i in range(ntracesTOT) :
        visible.append(False)
    
    visible[0] = True

    visible_a = []

    for i in range(ntraces_a) :
        visible_a.append(True)

    visible_b = []

    for i in range(ntraces_b) :
        visible_b.append(True)

    visibility_matrix = np.empty((4, len(visible)), dtype=bool)
    for i in range(4):
        visibility_matrix[i,:] = visible

    for i in range(0, 2):
        j = len(visible_a)*i
        visibility_matrix[i,j+1:j+1+len(visible_a)] = visible_a

    last_index = j + len(visible_a)

    for i in range(0, 2):
        j = len(visible_b)*i + last_index
        visibility_matrix[i+2,j+1:j+1+len(visible_b)] = visible_b

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["visible", visibility_matrix[0]],
                        label="Color: epochs   |   Symbol: batch_size",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[1]],
                        label="Color: epochs   |  Symbol: batch_size   |   Size: L1_dropout_rate",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[2]],
                        label="Color: batch_size   |  Symbol: epochs",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[3]],
                        label="Color: batch_size   |  Symbol: epochs   |   Size: L1_dropout_rate",
                        method="restyle"
                    ),

                ]),
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.17,
                yanchor="top"
            ),
        ]
    )

    fig.update_layout(
        annotations=[
            dict(text="Customize:", x=0, xref="paper", y=1.15, yref="paper",
                align="left", showarrow=False),
                 ])
    
    fig.update_traces(line=dict(color='Darkslategray', width=5), marker=dict(line=dict(width=1.2, color='DarkSlateGrey')))

    fig.update_yaxes(title_font=dict(size=20))
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_layout(xaxis = dict(tickfont = dict(size=15)), yaxis = dict(tickfont = dict(size=15)),
                      xaxis_title_text='Accuracy', yaxis_title_text=name, legend_title_text='Configuration', coloraxis_colorbar=dict(yanchor="top", y=1, x=-0.4,
                                          ticks="outside"))

    st.plotly_chart(fig)

    return df