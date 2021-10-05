import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics

# ---Part 1: Build project-------------------------------------------------------
data = pd.read_csv('train.csv')

# Data preprocessing
data['Sex'] = data['Sex'].map(lambda x: 0 if x == 'male' else 1)
data = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']]
data = data.dropna()

X = data.drop(['Survived'], axis=1)
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale data
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# Build model
model = LogisticRegression()
model.fit(train_features, y_train)

# Evaluation
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_predict = model.predict(test_features)
confusion = metrics.confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]
metrics.classification_report(y_test, y_predict)

# Caculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)

# Caculate AUC
auc = metrics.roc_auc_score(y_test, y_predict)
print('AUC:', auc)

# ---Part 2: Show project result with Streamlit------------------------------------------
st.title('Data Science')
st.header('Titanic Survival Predict Project')

# Tạo menu
menu = ['Overview', 'Build Projection', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Overview':
# Hiển thị giới thiệu chung
    st.subheader("Overview")
    st.write("""
    #### The data has been split into two groups:
    - training set (train.csv):
    The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
    - test set (test.csv):
    The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
    - gender_submission.csv:  a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
    """)

elif choice == 'Build Projection':
# Xây dựng mô hình
    # Hiển thị dữ liệu
    st.subheader('Build Projection')
    st.write('### Data Processing')
    st.write('#### Show Data')
    st.table(data.head(5))

    # Tham số đánh giá
    st.write('#### Build model and evaluation')
    st.write('Train set score: {}'.format(round(train_score,2)))
    st.write('Test set score: {}'.format(round(test_score,2)))
    st.write('Confussion matrix:')
    st.table(confusion)
    st.write(metrics.classification_report(y_test, y_predict))
    st.write('#### AUC: %.3f' %auc)

    # Trực quan hóa kết quả
    st.write('#### Visualization')
    fig, ax = plt.subplots()
    ax.bar(['False Nagative', 'True Negative', 'True Positive', 'False Positive'], [FN, TN, TP, FP])
    st.pyplot(fig)
    # roc curve
    st.write('ROC curve')
    fig1, ax1 = plt.subplots()
    ax1.plot([0,1],[0,1], linestyle='--')
    ax1.plot(fpr, tpr, marker='.')
    ax1.set_title('ROC curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    st.pyplot(fig1)

elif choice == 'New Prediction':
    st.subheader("Make New Prediction")
    st.write("#### Input / Select Data")
    name = st.text_input('Name of Passenger:')
    sex = st.radio('Sex', options=['Male', 'Female'])
    age = st.slider('Age', 1, 100, 1)
    Pclass = np.sort(data['Pclass'].unique())
    pclass = st.selectbox('Pclass',  options=Pclass)
    max_sibsp = max(data['SibSp'])
    sibsp = st.slider('SibSp', 0, max_sibsp, 1)
    max_parch = max(data['Parch'])
    parch = st.slider('Parch', 0, max_parch, 1)
    max_fare = round(max(data['Fare'])+10,2)
    fare = st.slider('Fare', 0.0, max_fare, 0.1)

    # Make new prediction
    sex = 0 if sex == 'Male' else 1
    new_data = scaler.transform([[sex, age, pclass, sibsp, parch, fare]])
    prediction = model.predict(new_data)
    predict_probability = model.predict_proba(new_data)

    if prediction[0] == 1:
        	st.subheader('Passenger {} would have survived with a probability of {}%'.format(name, 
                                                    round(predict_probability[0][1]*100 , 2)))
    else:
	    st.subheader('Passenger {} would not have survived with a probability of {}%'.format(name, 
                                                    round(predict_probability[0][0]*100 , 2)))
