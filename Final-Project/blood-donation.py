import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Blood Donation Prediction app")
    st.sidebar.title("Blood Donation Prediction app")
    st.markdown("Chances of a person donated blood in a particular month")
    st.markdown('Blood transfusion saves lives - from replacing lost blood during major surgery or a serious injury to treating various illnesses and blood disorders. Ensuring that there\'s enough blood in supply whenever needed is a serious challenge for the health professionals.')
    st.markdown('According to WebMD, "about 5 million Americans need a blood transfusion every year."Our dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive. We want to predict whether or not a donor will give blood the next time the vehicle comes to campus.The data is stored in datasets/transfusion.data and it is structured according to RFMTC marketing model (a variation of RFM).')
    st.sidebar.markdown("Chances of a person donated blood in a particular month")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("C:\\Users\\DELL PC\\Music\\Blood_donation_prediction\\transfusion\\transfusion.data")
        labelencoder=LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        df.rename(columns={'whether he/she donated blood in March 2007': 'target'},inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'),df.target
        ,test_size=0.25,random_state=42,stratify=df.target)
        return X_train, X_test, y_train, y_test

    def log_normalize(df):
        X_train, X_test, y_train, y_test = split(df)
        X_train_normed, X_test_normed = X_train.copy(), X_test.copy()
        # Specify which column to normalize
        col_to_normalize = X_train_normed.var().idxmax(axis=1)
        # Log normalization
        for df_ in [X_train_normed, X_test_normed]:
            # Add log normalized column
            df_['monetary_log'] = df_[col_to_normalize]
            # Drop the original column
            df_.drop(columns=col_to_normalize, inplace=True)
        return X_train_normed,X_test_normed


    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            #fig, ax = matplotlib.pyplot.subplots()
            #ax.plot([0,0.5,1],[0,0.5,1])
            #st.pyplot(fig)

         
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

    df = load_data()
    class_names = ['Donated', 'Not Donated']
    
    x_train, x_test, y_train, y_test = split(df)
    x_train_normed,x_test_normed = log_normalize(df)

    st.sidebar.subheader("Logistic Regression")
    #classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","TPOT"))
    classifier = "Logistic Regression"

    if st.sidebar.checkbox("Show raw data", True):
        st.subheader("Blood Donation Prediction Dataset")
        st.write(df)
        #st.markdown("This data set includes descriptions of bank account credit card payment data, we prdict the result analyzing these datas.")

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.05, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter,solver='liblinear',random_state=42)
            model.fit(x_train_normed, y_train)
            accuracy = model.score(x_test_normed, y_test)
            y_pred = model.predict(x_test_normed)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


if __name__ == '__main__':
    main()