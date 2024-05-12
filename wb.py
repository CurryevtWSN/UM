#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Uveal Melanoma Distant Metastasis Prediction System - A Retrospective Observational Study Based on Machine Learning')
st.title('Uveal Melanoma Distant Metastasis Prediction System - A Retrospective Observational Study Based on Machine Learning')

#%%set variables selection
st.sidebar.markdown('## Variables Selection')
Grade_Recode =  st.sidebar.selectbox("Grade recode",('Grade I','Grade II','Grade III',"Grade IV"),index=0)
Diagnostic_Confirmation = st.sidebar.selectbox("Diagnostic confirmation methods",
                                               ('Positive histology',
                                                'Radiography without microscopic confirm',
                                                'Direct visualization without microscopic confirmation',
                                                "Positive exfoliative cytology, no positive histology",
                                                "Clinical diagnosis only",
                                                "Others"),index=0)
Age =  st.sidebar.selectbox("Age",('< 20 years','20-40 years','40-60 years',"60-80 years","> 80 years"),index=3)
Laterality = st.sidebar.selectbox("Laterality",('Left','Right'),index=0)
Primary_Site = st.sidebar.selectbox("Primary site",('Choroid','Ciliary body'),index=0)
Months_from_diagnosis_to_treatment = st.sidebar.slider("Months from diagnosis to treatment (months)",0,20,value = 4,step = 1)
Rural_Urban_Continuum_Code = st.sidebar.selectbox("Rural urban continuum code",('<250000 population','250000-1 million population', '≥1 million population', 'Others'),index=0)
Radiation_recode = st.sidebar.selectbox("Radiation recode",('No','Yes'),index=0)
Total_number_of_malignant_tumors = st.sidebar.slider("Total number of malignant tumors",0,10,value = 2,step = 1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Xiamen university')
#传入数据
map = {'< 20 years':0,
       '20-40 years':1,
       '40-60 years':2,
       "60-80 years":3,
       "> 80 years":4,
       'Grade I':1,
       'Grade II':2,
       "Grade III":3,
       "Grade IV":4,
       "Positive histology":1,
       'Radiography without microscopic confirm':2,
       'Direct visualization without microscopic confirmation':3,
       'Positive exfoliative cytology, no positive histology':4,
       'Clinical diagnosis only':5,
       'Others':6,
       'Left':1,
       'Right':2,
       'Choroid':1,
       'Ciliary body':2,
       '<250000 population':1,
       '250000-1 million population':2, 
       '≥1 million population':3, 
       'Others':4,
       'No':0,
       'Yes':1}
Grade_Recode =map[Grade_Recode]
Diagnostic_Confirmation = map[Diagnostic_Confirmation]
Age = map[Age]
Laterality = map[Laterality]
Primary_Site = map[Primary_Site]
Rural_Urban_Continuum_Code = map[Rural_Urban_Continuum_Code]
Radiation_recode = map[Radiation_recode]
# 数据读取，特征标注
#%%load model
mlp_model = joblib.load('mlp_model.pkl')

#%%load data
hp_train = pd.read_csv('oversampled_data.csv')
features = ['Grade_Recode',
            'Diagnostic_Confirmation',
            'Primary_Site',
            'Age',
            'Rural_Urban_Continuum_Code',
            'Laterality',
            'Total_number_of_malignant_tumors',
            'Radiation_recode',
            'Months_from_diagnosis_to_treatment']
target = "Distant_metastasis"
y = np.array(hp_train[target])
sp = 0.5

is_t = (mlp_model.predict_proba(np.array([[Grade_Recode,Diagnostic_Confirmation,Primary_Site,Age,Rural_Urban_Continuum_Code,Laterality,Total_number_of_malignant_tumors,Radiation_recode,Months_from_diagnosis_to_treatment]]))[0][1])> sp
prob = (mlp_model.predict_proba(np.array([[Grade_Recode,Diagnostic_Confirmation,Primary_Site,Age,Rural_Urban_Continuum_Code,Laterality,Total_number_of_malignant_tumors,Radiation_recode,Months_from_diagnosis_to_treatment]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probability of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[Grade_Recode,Diagnostic_Confirmation,Primary_Site,Age,Rural_Urban_Continuum_Code,Laterality,Total_number_of_malignant_tumors,Radiation_recode,Months_from_diagnosis_to_treatment]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = mlp_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of MLP model')
    fig, ax = plt.subplots(figsize=(12, 6))
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of MLP model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of MLP model')
    mlp_prob = mlp_model.predict(X)
    cm = confusion_matrix(y, mlp_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of MLP model")
    disp1 = plt.show()
    st.pyplot(disp1)

