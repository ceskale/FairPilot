import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def SingleModelParetoLR(dfPareto, fmPareto, fm1, sem_fm1, fm2, sem_fm2, fm3, sem_fm3, fm4, sem_fm4, fm5, sem_fm5, fm6, sem_fm6,
                        mts, ets,
                        penalty, C, df_tot, name) :

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

    pareto_penalty = []
    pareto_C = []

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
        pareto_penalty.append(penalty[index2[0,0]])
        pareto_C.append(C[index2[0,0]])
    

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
        'penalty': pareto_penalty,
        'C': pareto_C,
    }

    df = pd.DataFrame(data=solutions_space)

    logC = np.log10(df_tot['C']) + 4

    fig1 = px.line(df, x="Accuracy", y=name, hover_data=['penalty','C'], markers=True)
    fig1.update_traces(line_shape="vh")
    fig2 = px.scatter(df_tot, x = "Accuracy", y = name, size=logC, symbol='penalty',
                    hover_data=['penalty','C'])
    fig3 = px.scatter(df_tot, x = "Accuracy", y = name, color=logC, symbol='penalty',
                hover_data=['penalty','C'])

    ntraces = df_tot['penalty'].nunique()

    fig = go.Figure()
    fig.add_trace(go.Scatter(fig1.data[0], visible=True))

    for i in range(ntraces):
        fig.add_trace(go.Scatter(fig2.data[i], visible=True))

    for i in range(ntraces):
        fig.add_trace(go.Scatter(fig3.data[i], visible=False))

    ntracesTOT = 1 + ntraces * 2

    visible = []

    for i in range(ntracesTOT) :
        visible.append(False)
    
    visible[0] = True

    visible_vec = []
    
    for i in range(ntraces) :
        visible_vec.append(True)

    visibility_matrix = np.empty((2, len(visible)), dtype=bool)
    for i in range(2):
        visibility_matrix[i,:] = visible

    for i in range(2):
        j = len(visible_vec)*i
        visibility_matrix[i,j+1:j+1+len(visible_vec)] = visible_vec

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["visible", visibility_matrix[0]],
                        label="Size: C   |  Symbol: penalty",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[1]],
                        label="Color: C   |   Symbol: penalty",
                        method="restyle"
                    )
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