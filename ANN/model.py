import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Customer_Churn.csv', index_col=0)
print('No of rows: ',len(df))
# df.info()

scaler = MinMaxScaler()
df[['tenure']] = scaler.fit_transform(df[['tenure']])
df[['MonthlyCharges']] = scaler.fit_transform(df[['MonthlyCharges']])
df[['TotalCharges']] = scaler.fit_transform(df[['TotalCharges']])

ordinal_encoder = OrdinalEncoder(categories=[['Male','Female']])
df['gender'] = ordinal_encoder.fit_transform(df[['gender']])

ordinal_encoder = OrdinalEncoder(categories=[['Yes','No']])
df['Partner'] = ordinal_encoder.fit_transform(df[['Partner']])
df['Dependents'] = ordinal_encoder.fit_transform(df[['Dependents']])
df['PhoneService'] = ordinal_encoder.fit_transform(df[['PhoneService']])
df['PaperlessBilling'] = ordinal_encoder.fit_transform(df[['PaperlessBilling']])

one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

df = pd.concat([
    df.drop(columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']),
    one_hot_encoder.fit_transform(df[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']])
    ],axis=1)

df.to_csv('data_normalized.csv')

x = df.drop(['Churn'],axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print('No of Training Rows: ',len(x_train))
print('No of Testing Rows:',len(x_test))

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,),max_iter=1000)

mlp_classifier.fit(x_train,y_train)

predict = mlp_classifier.predict(x_test)

print('Hidden Layer Size:',mlp_classifier.hidden_layer_sizes)
print('Number of Layers:',mlp_classifier.n_layers_)
print('Number of Iterations:',mlp_classifier.n_iter_)
print('Classes:',mlp_classifier.classes_)
print('Accuracy:',accuracy_score(y_test, predict))
print(classification_report(y_test, predict))