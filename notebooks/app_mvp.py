import streamlit as st
import altair as alt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools
import datetime as dt
from tools import dataoveriew, plot_roc_curve, plot_confusion_matrix
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

def feature_engineering_data(data, fecha):
    
    data['Born Date'] = data['Born Date'].replace(np.nan, datetime(1970, 1, 1))
    data['Edad'] = 0
    fecha = datetime(2021, 4, 18)

    for i in range(len(data['Born Date'])):
        if data.loc[i, 'Status'] == 0:
            data.loc[i,'Edad'] = ((fecha - data.loc[i,'Born Date']).days)/365
        else:
            data.loc[i,'Edad'] = ((data.loc[i,'Status Date'] - data.loc[i,'Born Date']).days)/365
    
    for i in range(len(data['Edad'])):
        if data.loc[i, 'Edad'] < 18:
            data.loc[i, 'Edad'] = data['Edad'].mean()
        else:
            continue
        
    # Mapping Edad 
    data.loc[data['Edad'] <= 30, 'Rango_Edad'] = "18-30"
    data.loc[(data['Edad'] > 30) & (data['Edad'] <= 40), 'Rango_Edad'] = "30-40"
    data.loc[(data['Edad'] > 40) & (data['Edad'] <= 50), 'Rango_Edad'] = "40-50"
    data.loc[(data['Edad'] > 50) & (data['Edad'] <= 60), 'Rango_Edad'] = "50-60"
    data.loc[(data['Edad'] > 60) & (data['Edad'] <= 70), 'Rango_Edad'] = "60-70"
    data.loc[(data['Edad'] > 70) & (data['Edad'] <= 80), 'Rango_Edad'] = "70-80"
    data.loc[data['Edad'] > 80, 'Rango_Edad'] = "+80"
    
    # Mapping Income Amount
    data.loc[data['Income Amount'] <= 1000, 'Income'] = "0-1000"
    data.loc[(data['Income Amount'] > 1000) & (data['Income Amount'] <= 1500), 'Income'] = "1000-1500"
    data.loc[(data['Income Amount'] > 1500) & (data['Income Amount'] <= 2000), 'Income'] = "1500-2000"
    data.loc[(data['Income Amount'] > 2000) & (data['Income Amount'] <= 3000), 'Income'] = "2000-3000"
    data.loc[data['Income Amount'] > 3000, 'Income'] = "+3000"
    
    # Días hasta el 18/04/2021 si OP y hasta cuando se dio de baja si BAJA
    data['Dias_Activo'] = 0
#    fecha = datetime(2021, 4, 18)
    
    for i in range(len(data['Start Date'])):
        if data.loc[i, 'Status'] == 0:
            data.loc[i,'Dias_Activo'] = (fecha - data.loc[i,'Start Date']).days
        else:
            data.loc[i,'Dias_Activo'] = (data.loc[i,'Status Date'] - data.loc[i,'Start Date']).days
    
    return data


def show_countplot(data):

    st.subheader("Data Visualization")
    feature_x = st.selectbox("Seleccionar variable para la X", ['Provincia','Gender', 'Housing Type', 'Property Type', 'Labor Situation', 'Marital Status', 
                       'Nationality', 'Rango_Edad', 'Income', 'Rango Precio', 'Number Pay'])

    feature_seg = st.selectbox("Seleccionar variable para segmentar", ['Gender', 'Housing Type', 'Property Type', 'Labor Situation', 'Marital Status', 
                      'Provincia', 'Nationality', 'Rango_Edad', 'Income', 'Rango Precio', 'Number Pay'])

    chart = alt.Chart(data).mark_bar().encode(alt.X(feature_x), y='count()', color = feature_seg).properties(width=800, height=500)
    st.altair_chart(chart)
    

def machine_learning_model(data, data_to_result):

    st.header("Machine Learning Models")

    filename1 = '../mvp_pkl/dias_activo_sca.pkl'
    scaler1 = pickle.load(open(filename1, 'rb'))
    data['Dias_Activo_sca'] = scaler1.transform(data['Dias_Activo'].values.reshape(-1, 1))
    
    filename2 = '../mvp_pkl/quejas_sca.pkl'
    scaler2 = pickle.load(open(filename2, 'rb'))
    data['Quejas_sca'] = scaler2.transform(data['Quejas'].values.reshape(-1, 1))
    
    filename3 = '../mvp_pkl/incidencias_sca.pkl'
    scaler3 = pickle.load(open(filename3, 'rb'))
    data['Incidencias_sca'] = scaler3.fit_transform(data['Incidencias'].values.reshape(-1, 1))

    data_filtered = data[['Gender', 'Housing Type', 'Property Type', 'Labor Situation', 'Marital Status',
                      'Provincia', 'Nationality', 'Rango_Edad', 'Income', 'Rango Precio', 'Number Pay', 'Dias_Activo_sca',
                      'Quejas_sca', 'Incidencias_sca', 'Status']]    


    
    X = data_filtered.drop(['Status'],axis=1)
    y = data_filtered['Status']

    # Target encoder
    filename = '../mvp_pkl/TE_encoder.pkl'
    TE_encoder = pickle.load(open(filename, 'rb'))
    X = TE_encoder.transform(X)

    algoritmos = ["Decision Tree", "Logistic Regression", "Random Forest"]
    classifier = st.selectbox("Seleccionar algoritmo", algoritmos)

    if classifier == "Decision Tree":
        filename = '../mvp_pkl/DT_model.pkl'
        DT = pickle.load(open(filename, 'rb'))        
        y_pred = DT.predict(X)
        result = DT.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = DT.score(X, y)
        
    elif classifier == "Logistic Regression":
        filename = '../mvp_pkl/LR_model.pkl'
        LR = pickle.load(open(filename, 'rb'))        
        y_pred = LR.predict(X)
        result = LR.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = LR.score(X, y)
        
    elif classifier == "Random Forest":
        filename = '../mvp_pkl/rfc_model.pkl'
        rfc = pickle.load(open(filename, 'rb'))       
        y_pred = rfc.predict(X)
        result = rfc.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = rfc.score(X, y)
    
    else:
        raise NotImplementedError()

    class_names = ['Activo','Baja']

    st.subheader("Metrics")
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(y, y_pred, labels=class_names).round(2))
    st.write("Recall: ", recall_score(y, y_pred, labels=class_names).round(2))

    st.subheader("Confusion Matrix")

    # Desactivar warning:
    st.set_option('deprecation.showPyplotGlobalUse', False)

    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
    st.pyplot()
#    st.write("Confusion matrix: ", cm)  
   
    proba_baja = pd.DataFrame(result, columns = ['Probabilidad de baja'])
    resultado = pd.concat([data_to_result, proba_baja], axis = 1)

    st.header("Customers with the highest likelihood of churning:")
    
    threshold = st.slider('Mostar clientes con probabilidad de baja superior a (%): ', 
                          min_value=0.0, max_value=100.0, value = 70.0, step = 0.5)
    resultado_filtrado = resultado[(resultado['Probabilidad de baja'] >= threshold/100)]
    resultado_ordenado = resultado_filtrado.sort_values('Probabilidad de baja',ascending=False)
    
    st.write("Treshold seleccionado (%): ", threshold)
    st.write("Número de clientes mostrados: ", resultado_ordenado.shape[0])    
    st.write(resultado_ordenado)

def main():
    
    st.title("Customer Churn Prediction app")
    
    uploaded_file = st.file_uploader("Subir archivo .xlsx", type="xlsx")
    
    if not uploaded_file:
        st.warning('Por favor, sube un fichero')
        st.stop()
    
    elif uploaded_file:
        data = pd.read_excel(uploaded_file)
    
#        st.table(data)


    fecha = st.date_input("Fecha de la extracción", value = datetime(2021, 4, 18))
    
    st.header("Data Exploration")
        
#    data = pd.read_excel('../data/test_com_valencia.xlsx')

   
    st.subheader("Source Data")
    st.write("Número de clientes en el archivo: ", data.shape[0]) 
    if st.checkbox("Show Source Data"):
        st.write(data)
   
    feature_engineering_data(data, fecha)
    data_to_result = data.copy()
    show_countplot(data)
    machine_learning_model(data, data_to_result)

main()