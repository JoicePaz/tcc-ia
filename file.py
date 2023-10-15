import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.express as px
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import sciki
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



file_path ='/cert_2010-2019.csv'
dataset = pd.read_csv(file_path, sep = ";")

display(dataset.head())

dataset.info()

#sample the dataset

df = pd.DataFrame(dataset)

# Set the display option for float format
pd.set_option('display.float_format', '{:.2f}'.format)

# Generate and print the descriptive statistics
df.describe()

dataset.info()

dataset.shape

dataset = dataset.drop(['Total', 'Outros'], axis=1)
print(dataset)

#sample the dataset
df = pd.DataFrame(dataset)

# Set the display option for float format
pd.set_option('display.float_format', '{:.2f}'.format)

# Generate and print the descriptive statistics
df.describe()

dataset['Ano'].value_counts()

dataset.isnull().sum()

def check_mes(Mes):
    if Mes == "Janeiro":
        nMes = 1
    elif Mes == "Fevereiro":
        nMes = 2
    elif Mes == "Março":
        nMes = 3
    elif Mes == "Abril":
        nMes = 4
    elif Mes == "Maio":
        nMes = 5
    elif Mes == "Junho":
        nMes = 6
    elif Mes == "Julho":
        nMes = 7
    elif Mes == "Agosto":
        nMes = 8
    elif Mes == "Setembro":
        nMes = 9
    elif Mes == "Outubro":
        nMes = 10
    elif Mes == "Novembro":
        nMes = 11
    else:
        nMes = 12
    return nMes

dataset['NMes'] = dataset['Mes'].apply(check_mes)

dataset['data'] = dataset['Ano'].map(str)+'-'+dataset['NMes'].map(str)+'-01'
dataset.head()

dataset['data']= dataset['data'].astype('datetime64[ns]')
dataset = dataset.sort_values(by='data')
dataset.head()

br_incidentes = go.Figure()
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['Worm'], name="Worm", stackgroup='one'))
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['DOS'], name="DOS", stackgroup='one'))
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['Invasao'], name="Invasao", stackgroup='one'))
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['Web'], name="Web", stackgroup='one'))
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['Scan'], name="Scan", stackgroup='one'))
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['Fraude'], name="Fraude", stackgroup='one'))
br_incidentes.add_trace(go.Scatter(x=dataset['data'], y=dataset['Outros'], name="Outros", stackgroup='one'))
br_incidentes.update_layout(
   title="<b>Evolução dos incidentes de internet no Brasil entre 2010 e 2019 <b>",
   yaxis_title="N.º de incidentes"
   )
br_incidentes.update_xaxes(title="<b>Período<b>")
br_incidentes.update_yaxes(title="<b>Quantidade<b>")
br_incidentes.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)


br_incidentes.show()

br_ciclo_anual_incidentes = go.Figure()
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['Worm'], name="Worm"))
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['DOS'], name="DOS"))
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['Invasao'], name="Invasão"))
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['Web'], name="Web"))
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['Scan'], name="Scan"))
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['Fraude'], name="Fraude"))
br_ciclo_anual_incidentes.add_trace(go.Bar(x=dataset['Mes'], y=dataset['Outros'], name="Outros"))
br_ciclo_anual_incidentes.update_layout(
   title="<b>Ciclo anual de incidentes<b>",
   yaxis_title="<b>N.º de incidentes<b>"
   )
br_ciclo_anual_incidentes.update_xaxes(title="<b>Mês<b>")
br_ciclo_anual_incidentes.update_yaxes(title="<b>Quantidade<b>")
br_ciclo_anual_incidentes.show()


tot_anual_incidentes = go.Figure()
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['Worm'], name="Worm"))
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['DOS'], name="DOS"))
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['Invasao'], name="Invasão"))
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['Web'], name="Web"))
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['Scan'], name="Scan"))
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['Fraude'], name="Fraude"))
tot_anual_incidentes.add_trace(go.Bar(x=dataset['Ano'], y=dataset['Outros'], name="Outros"))
tot_anual_incidentes.update_layout(title="<b>Total de incidentes no período 2010 - 2019<b>")
tot_anual_incidentes.update_traces(hoverinfo='x+y')
tot_anual_incidentes.show()


dist_prob_incidentes = make_subplots(rows=1,
                                    cols=7,
                                    specs=[[{'type':'box'}, {'type':'box'}, {'type':'box'}, {'type':'box'}, {'type':'box'}, {'type':'box'}, {'type':'box'}]])
dist_prob_incidentes.add_trace(go.Box(y=dataset['Worm'], name="Worm"),1,1)
dist_prob_incidentes.add_trace(go.Box(y=dataset['DOS'], name="DOS"),1,2)
dist_prob_incidentes.add_trace(go.Box(y=dataset['Invasao'], name="Invasão"),1,3)
dist_prob_incidentes.add_trace(go.Box(y=dataset['Web'], name="Web"),1,4)
dist_prob_incidentes.add_trace(go.Box(y=dataset['Scan'], name="Scan"),1,5)
dist_prob_incidentes.add_trace(go.Box(y=dataset['Fraude'], name="Fraude"),1,6)
dist_prob_incidentes.add_trace(go.Box(y=dataset['Outros'], name="Outros"),1,7)
dist_prob_incidentes.update_layout(title="<b>Distribuições de probabilidade de incidentes de TI no Brasil no período 2010 - 2019<b>",
                                   images=[dict(source='', xref="paper", yref="paper", x=0, y=1, sizex=0.12, sizey=0.12, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3),
                                           dict(source='', xref="paper", yref="paper", x=0.145, y=1, sizex=0.12, sizey=0.2, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3),
                                           dict(source='', xref="paper", yref="paper", x=0.29, y=1, sizex=0.12, sizey=0.2, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3),
                                           dict(source='', xref="paper", yref="paper", x=0.44, y=1, sizex=0.12, sizey=0.25, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3),
                                           dict(source='', xref="paper", yref="paper", x=0.59, y=1, sizex=0.12, sizey=0.25, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3),
                                           dict(source='', xref="paper", yref="paper", x=0.73, y=1, sizex=0.12, sizey=0.25, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3),
                                           dict(source='', xref="paper", yref="paper", x=0.88, y=1, sizex=0.12, sizey=0.25, layer="below", xanchor="left", yanchor="top", sizing="stretch", opacity=0.3)
                                           ])

dist_prob_incidentes.show()


#split dataset in features and target variable
X = dataset.drop(['Ano'], axis=1)
y = dataset['Ano']

# split X and y into training and testing sets 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

#feature engeenering
X_train.dtypes

X_train.head()

X_test.head()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Mes'])
    ],
    remainder='passthrough'
)

X_train_transformed = preprocessor.fit_transform(X_train)

# Get the feature names after one-hot encoding
feature_names = list(preprocessor.named_transformers_['onehot'].get_feature_names(['Mes'])) + list(X.columns[1:])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc

fig, ax = plt.subplots()
plot_confusion_matrix(pipeline, X_test, y_test, cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusão')
plt.show()

# Curva ROC
probas_ = pipeline.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
