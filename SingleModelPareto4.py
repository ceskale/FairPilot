import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def SingleModelParetoRF(dfPareto, fmPareto, fm1, sem_fm1, fm2, sem_fm2, fm3, sem_fm3, fm4, sem_fm4, fm5, sem_fm5, fm6, sem_fm6,
                        mts, ets,
                        min_samples_leaf, min_samples_split, max_features, criterion, class_weight, bootstrap, max_depth, max_samples, df_tot, name) :

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

    pareto_min_samples_leaf = []
    pareto_min_samples_split = []
    pareto_max_features = []
    pareto_criterion = []
    pareto_class_weight = []
    pareto_bootstrap = []
    pareto_max_depth = []
    pareto_max_samples = []

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
        pareto_min_samples_leaf.append(min_samples_leaf[index2[0,0]])
        pareto_min_samples_split.append(min_samples_split[index2[0,0]])
        pareto_max_features.append(max_features[index2[0,0]])
        pareto_criterion.append(criterion[index2[0,0]])
        pareto_class_weight.append(class_weight[index2[0,0]])
        pareto_bootstrap.append(bootstrap[index2[0,0]])
        pareto_max_depth.append(max_depth[index2[0,0]])
        pareto_max_samples.append(max_samples[index2[0,0]])
    

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
        'min_samples_leaf': pareto_min_samples_leaf,
        'min_samples_split': pareto_min_samples_split,
        'max_depth': pareto_max_depth,
        'max_features': pareto_max_features,
        'criterion': pareto_criterion,
        'class weight': pareto_class_weight,
        'bootstrap': pareto_bootstrap,
        'max_samples': pareto_max_samples,
    }

    df = pd.DataFrame(data=solutions_space)

    st.write(df)

    fig1 = px.line(df, x="Accuracy", y=name, hover_data=['min_samples_leaf','min_samples_split', 'max_depth', 'criterion','class weight','max_features', 'bootstrap', 'max_samples'], markers=True)
    fig1.update_traces(line_shape="vh")
    fig2 = px.scatter(df_tot, x = "Accuracy", y = name, color='criterion', size='min_samples_split', symbol='class_weight',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig3 = px.scatter(df_tot, x = "Accuracy", y = name, color='criterion', size='min_samples_leaf', symbol='class_weight',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig4 = px.scatter(df_tot, x = "Accuracy", y = name, color='class_weight', size='min_samples_split', symbol='criterion',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig5 = px.scatter(df_tot, x = "Accuracy", y = name, color='class_weight', size='min_samples_leaf', symbol='criterion',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig6 = px.scatter(df_tot, x = "Accuracy", y = name, color='max_features', size='min_samples_split', symbol='class_weight',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig7 = px.scatter(df_tot, x = "Accuracy", y = name, color='max_features', size='min_samples_leaf', symbol='class_weight',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig8 = px.scatter(df_tot, x = "Accuracy", y = name, color='bootstrap', size='min_samples_split', symbol='class_weight',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])
    fig9 = px.scatter(df_tot, x = "Accuracy", y = name, color='bootstrap', size='min_samples_leaf', symbol='class_weight',
                    hover_data=['min_samples_leaf','min_samples_split','criterion','class_weight','max_features','bootstrap'])    

    ntraces_a = df_tot['criterion'].nunique() * df_tot['class_weight'].nunique()
    ntraces_b = df_tot['max_features'].nunique() * df_tot['class_weight'].nunique()
    ntraces_c = df_tot['bootstrap'].nunique() * df_tot['class_weight'].nunique()

    fig = go.Figure()
    fig.add_trace(go.Scatter(fig1.data[0], visible=True))

    for i in range(ntraces_a):
        fig.add_trace(go.Scatter(fig2.data[i], visible=True))

    for i in range(ntraces_a):
        fig.add_trace(go.Scatter(fig3.data[i], visible=False))

    for i in range(ntraces_a):
        fig.add_trace(go.Scatter(fig4.data[i], visible=False))

    for i in range(ntraces_a):
        fig.add_trace(go.Scatter(fig5.data[i], visible=False))

    for i in range(ntraces_b):
        fig.add_trace(go.Scatter(fig6.data[i], visible=False))

    for i in range(ntraces_b):
        fig.add_trace(go.Scatter(fig7.data[i], visible=False))

    for i in range(ntraces_c):
        fig.add_trace(go.Scatter(fig8.data[i], visible=False))

    for i in range(ntraces_c):
        fig.add_trace(go.Scatter(fig9.data[i], visible=False))

    ntracesTOT = 1 + ntraces_a * 4 + ntraces_b * 2 + ntraces_c * 2

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

    visible_c = []

    for i in range(ntraces_c) :
        visible_c.append(True)

    visibility_matrix = np.empty((8, len(visible)), dtype=bool)
    for i in range(8):
        visibility_matrix[i,:] = visible

    for i in range(0, 4):
        j = len(visible_a)*i
        visibility_matrix[i,j+1:j+1+len(visible_a)] = visible_a

    last_index = j + len(visible_a)

    for i in range(0, 2):
        j = len(visible_b)*i + last_index
        visibility_matrix[i+4,j+1:j+1+len(visible_b)] = visible_b

    last_index = j + len(visible_b)

    for i in range(0, 2):
        j = len(visible_c)*i + last_index
        visibility_matrix[i+6,j+1:j+1+len(visible_c)] = visible_c

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["visible", visibility_matrix[0]],
                        label="Color: criterion   |   Symbol: class_weight   |   Size: min_samples_split",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[1]],
                        label="Color: criterion   |  Symbol: class_weight   |   Size: min_samples_leaf",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[2]],
                        label="Color: class_weight   |  Symbol: criterion   |   Size: min_samples_split",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[3]],
                        label="Color: class_weight   |  Symbol: criterion   |   Size: min_samples_leaf",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[4]],
                        label="Color: max_features   |  Symbol: class_weight   |   Size: min_samples_split",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[5]],
                        label="Color: max_features   |  Symbol: class_weight   |   Size: min_samples_leaf",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[6]],
                        label="Color: bootstrap   |  Symbol: class_weight   |   Size: min_samples_split",
                        method="restyle"
                    ),
                    dict(
                        args=["visible", visibility_matrix[7]],
                        label="Color: boostrap   |  Symbol: class_weight   |   Size: min_samples_leaf",
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
    
    fig.update_traces(line=dict(color='Darkslategray', width=2), marker=dict(line=dict(width=1.2, color='DarkSlateGrey')))

    fig.update_yaxes(title_font=dict(size=20))
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_layout(xaxis = dict(tickfont = dict(size=15)), yaxis = dict(tickfont = dict(size=15)),
                      xaxis_title_text='Accuracy', yaxis_title_text=name, legend_title_text='Configuration')

    st.plotly_chart(fig)

    return df