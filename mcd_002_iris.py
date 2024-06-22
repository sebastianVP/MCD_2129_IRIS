import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# APLICACION IRIS PARA PREDICTION DE TIPO DE ESPECIES
         
Esta aplicacion predice el tipo de flor en base a sus mediciones de sepal y petal

""")
st.sidebar.header("Parametros de entrada por el Usuario")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('Sepal with',2.0,4.4,3.4)
    petal_length = st.sidebar.slider('Petal length',1.0,6.9,1.3)
    petal_width = st.sidebar.slider('Petal width',0.1,2.5,0.2)
    data= { 'sepal_length': sepal_length,
           'sepal_width': sepal_width,
           'petal_length': petal_length,
           'petal_width': petal_width}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

st.subheader("Parametros de entrada por el Usuario")
st.write(df)

iris = datasets.load_iris()
X= iris.data
Y= iris.target
clf =RandomForestClassifier()
clf.fit(X,Y)
##with open("modelo_iris",'wb') as output:
##  pickle.dump(clf,output)
#print("generar modelo")
#import os
#var= os.getcwd()
#new= os.path.join(var,'modelo_iris')
#clf = pickle.load(open(new,'rb'))

prediction =  clf.predict(df)
prediction_proba = clf.predict_proba(df)# probabilidad

st.subheader("Mostrando Etiquetas de Especies y su correspondiente Index")
df__ = pd.DataFrame(iris.target_names)
st.write(df__)


st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

