import streamlit as st
import altair as alt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools
import datetime as dt
import matplotlib.pyplot as plt
from tools import dataoveriew, plot_roc_curve, plot_confusion_matrix
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
from PIL import Image

def feature_engineering_data(data):
    
    data['Born Date'] = data['Born Date'].replace(np.nan, datetime(1970, 1, 1))

    data['Edad'] = ((data['Fecha Alta'] - data['Born Date']).dt.days)/365
    
    edad_mean = data['Edad'].mean()
    data['Edad'] = data['Edad'].apply(lambda x: edad_mean if x<18 else x)
        
    # Mapping Edad 
    data.loc[data['Edad'] <= 30, 'Rango_Edad'] = "18-30"
    data.loc[(data['Edad'] > 30) & (data['Edad'] <= 40), 'Rango_Edad'] = "30-40"
    data.loc[(data['Edad'] > 40) & (data['Edad'] <= 50), 'Rango_Edad'] = "40-50"
    data.loc[(data['Edad'] > 50) & (data['Edad'] <= 60), 'Rango_Edad'] = "50-60"
    data.loc[(data['Edad'] > 60) & (data['Edad'] <= 70), 'Rango_Edad'] = "60-70"
    data.loc[(data['Edad'] > 70) & (data['Edad'] <= 80), 'Rango_Edad'] = "70-80"
    data.loc[data['Edad'] > 80, 'Rango_Edad'] = "+80"
    
    # Mapping Income
    data.loc[data['Ingresos'] <= 1000, 'Income'] = "0-1000"
    data.loc[(data['Ingresos'] > 1000) & (data['Ingresos'] <= 1500), 'Income'] = "1000-1500"
    data.loc[(data['Ingresos'] > 1500) & (data['Ingresos'] <= 2000), 'Income'] = "1500-2000"
    data.loc[(data['Ingresos'] > 2000) & (data['Ingresos'] <= 3000), 'Income'] = "2000-3000"
    data.loc[data['Ingresos'] > 3000, 'Income'] = "+3000"
      
    return data


def show_countplot(data):

    st.subheader("Data Visualization")
    feature_x = st.selectbox("Select feature for X axis:", ['Tipo Inmueble', 'Tipo Propiedad','Gender', 'Provincia', 'Situacion Laboral', 'Estado Civil', 
                       'Pais', 'Rango_Edad', 'Income','Precio Total', 'Precio Contado', 'Pagos Anuales'])

    feature_seg = st.selectbox("Select feature to segment:", ['Precio Contado','Gender','Tipo Propiedad','Tipo Inmueble', 'Provincia', 'Situacion Laboral', 'Estado Civil', 
                       'Pais', 'Rango_Edad', 'Income', 'Pagos Anuales','Precio Total'])

    chart = alt.Chart(data).mark_bar().encode(alt.X(feature_x), y='count()', color = feature_seg).properties(width=800, height=500)
    st.altair_chart(chart)


def machine_learning_model(data, data_to_result):

    st.header("Machine Learning Models")

    filename1 = '../mvp_pkl/consumo_sca.pkl'
    scaler1 = pickle.load(open(filename1, 'rb'))
    data['Consumo_sca'] = scaler1.transform(data['Consumo Mes'].values.reshape(-1, 1))
    
    filename2 = '../pkl/quejas_sca.pkl'
    scaler2 = pickle.load(open(filename2, 'rb'))
    data['Quejas_sca'] = scaler2.transform(data['Quejas'].values.reshape(-1, 1))
    
    filename3 = '../pkl/incidencias_sca.pkl'
    scaler3 = pickle.load(open(filename3, 'rb'))
    data['Incidencias_sca'] = scaler3.fit_transform(data['Incidencias'].values.reshape(-1, 1))

    data_filtered = data[['Gender', 'Tipo Inmueble', 'Tipo Propiedad', 'Situacion Laboral', 'Estado Civil', 
                      'Provincia', 'Pais', 'Rango_Edad', 'Income', 'Precio Contado', 'Pagos Anuales', 'Precio Total',
                      'Quejas_sca', 'Incidencias_sca', 'Consumo_sca', 'Estado']]     
   
    X = data_filtered.drop(['Estado'],axis=1)
    y = data_filtered['Estado']

    # Target encoder
    filename = '../pkl/TE_encoder.pkl'
    TE_encoder = pickle.load(open(filename, 'rb'))
    X = TE_encoder.transform(X)

    algoritmos = ["Please choose an algorithm for prediction",
                  "Decision Tree", "Logistic Regression", "Random Forest", "Voting Classifier"]
    classifier = st.selectbox("Select algorithm:", algoritmos)

    if classifier == "Decision Tree":
        filename = '../pkl/DT_model.pkl'
        DT = pickle.load(open(filename, 'rb'))        
        y_pred = DT.predict(X)
        result = DT.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = DT.score(X, y)
        
    elif classifier == "Logistic Regression":
        filename = '../pkl/LR_model.pkl'
        LR = pickle.load(open(filename, 'rb'))        
        y_pred = LR.predict(X)
        result = LR.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = LR.score(X, y)
        
    elif classifier == "Random Forest":
        filename = '../pkl/rfc_model.pkl'
        rfc = pickle.load(open(filename, 'rb'))       
        y_pred = rfc.predict(X)
        result = rfc.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = rfc.score(X, y)
    
    elif classifier == "Voting Classifier":
        filename = '../pkl/voting_clf_model.pkl'
        voting = pickle.load(open(filename, 'rb'))       
        y_pred = voting.predict(X)
        result = voting.predict_proba(X)[:,1].reshape(-1, 1)
        accuracy = voting.score(X, y)
    
    else:
        st.stop()


    # Desactivar warning:
    st.set_option('deprecation.showPyplotGlobalUse', False)

    proba_baja = pd.DataFrame(result, columns = ['Churn_Probability'])
    resultado = pd.concat([data_to_result, proba_baja], axis = 1)
    st.write('')  
    
    st.header("Results Resume:")
    st.write("Churn % combining building type and upfront payment")
    resume = pd.crosstab(resultado['Tipo Inmueble'], resultado['Precio Contado'], values=resultado.Churn_Probability, aggfunc='mean').round(4)*100
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(resume, ax=ax, cmap=cmap, annot=True, cbar=False, linewidths=.5)
    st.write(fig)

    
    st.write('')  
    st.header("Customers with the highest likelihood of churning:")
    
    threshold = st.slider('Show customers with churn probability greater than (%): ', 
                          min_value=0.0, max_value=100.0, value = 70.0, step = 0.5)
    resultado_filtrado = resultado[(resultado['Churn_Probability'] >= threshold/100)]
    resultado_ordenado = resultado_filtrado.sort_values('Churn_Probability',ascending=False)
    
    st.write("Selected treshold (%): ", threshold)
    st.write("Number of customers showed: ", resultado_ordenado.shape[0])    
    st.write(resultado_ordenado)

    if st.button('Extract to .xlsx'):
        resultado_ordenado.to_excel("../data/prediction_result.xlsx", index = False)

# Main

# Set page config
apptitle = 'Customer Churn app'
st.set_page_config(page_title=apptitle, page_icon=":cry:")

header_pic = Image.open('../images/churn_icon.png')

st.title("Customer Churn Prediction app")

st.sidebar.image(header_pic, use_column_width=True)
st.sidebar.markdown("## Customer Churn Prediction app")
st.sidebar.markdown('When selecting the algorithm for the prediction, consider the following results:')
st.sidebar.info('🥇 Decision Tree Classifier')
st.sidebar.info('🥈 Voting Classifier')
st.sidebar.info('🥉 Logistic Regression')
st.sidebar.info('😢 Random Forest Classifier')

uploaded_file = st.file_uploader("Upload file .xlsx", type="xlsx")
      
if not uploaded_file:
    st.warning('Please upload a file')
    st.stop()
    
elif uploaded_file:
    data = pd.read_excel(uploaded_file)
    data['Quejas'] = data['Quejas'].replace(np.nan, 0).astype('int')
    data['Incidencias'] = data['Incidencias'].replace(np.nan, 0).astype('int')
    data['Cliente'] = data['Cliente'].astype('str')
    data['Estado'] = data['Estado'].astype('str').str.strip()
    data['Estado'] = data['Estado'].replace({'ACTIVO': 0, 'BAJA': 1}).astype(int)

st.header("Data Exploration")
   
st.subheader("Source Data")
st.write("Number of customers in the file: ", data.shape[0]) 
if st.checkbox("Show Source Data"):
    st.write(data)

feature_engineering_data(data)
data_to_result = data.copy()

show_countplot(data)
machine_learning_model(data, data_to_result)



