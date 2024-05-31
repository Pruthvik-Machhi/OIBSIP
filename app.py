import streamlit as st
import pandas as pd
import pickle
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

with open('model.pkl', 'rb') as model_file:
    clf_iris = pickle.load(model_file)

with open('stand.pkl', 'rb') as scaler_file:
    scaler_iris = pickle.load(scaler_file)

with open('model2xgb.pkl', 'rb') as model_file2:
    xgb = pickle.load(model_file2)

with open('minmax.pkl', 'rb') as scaler_file2:
    scaler2 = pickle.load(scaler_file2)

html_temp_title = """
    <div style="background-color:#ff0000;padding:10px;margin-bottom:20px">
    <h1 style="color:white;text-align:center;">OIBSIP Internship</h1>
    </div>
    """
st.markdown(html_temp_title, unsafe_allow_html=True)

html_attribution = """
    <div style="background-color:#28a745;padding:20px;margin-bottom:20px">
    <p style="color:white;text-align:center;font-size:22px;">Developed by Pruthvik Machhi</p>
    </div>
    """
st.markdown(html_attribution, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Iris Classification'):
        st.experimental_set_query_params(project="iris")

with col2:
    if st.button('Sales Prediction'):
        st.experimental_set_query_params(project="sales")

with col3:
    if st.button('SPAM Detection'):
        st.experimental_set_query_params(project="spam")

query_params = st.experimental_get_query_params()
project = query_params.get("project", ["iris"])[0]

if project == "iris":

    html_temp_subtitle = """
        <div style="background-color:#007bff;padding:10px;margin-bottom:20px">
        <h2 style="color:white;text-align:center;">Iris Flower Prediction</h2>
        </div>
        """
    st.markdown(html_temp_subtitle, unsafe_allow_html=True)

    def user_input_features():
        sepal_length = st.number_input('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.number_input('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.number_input('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.number_input('Petal width', 0.1, 2.5, 0.2)
        data = {'SepalLengthCm': sepal_length,
                'SepalWidthCm': sepal_width,
                'PetalLengthCm': petal_length,
                'PetalWidthCm': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    st.subheader('Enter Input ')
    df = user_input_features()

    st.subheader(' Input parameters')
    st.write(df)

    expected_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    df = df[expected_features]

    if st.button('Predict'):
        scaled_features = scaler_iris.transform(df)
        prediction = clf_iris.predict(scaled_features)
        species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        predicted_species = species[prediction[0]]
        st.subheader('Prediction')
        st.write(f"The predicted Iris species is: **{predicted_species}**")

elif project == "sales":
 
    html_temp_subtitle = """
        <div style="background-color:#007bff;padding:10px;margin-bottom:20px">
        <h2 style="color:white;text-align:center;">Sales Prediction</h2>
        </div>
        """
    st.markdown(html_temp_subtitle, unsafe_allow_html=True)

    def user_input_features():
        i1 = st.number_input('TV')
        i2 = st.number_input('Radio')
        i3 = st.number_input('Newspaper')
        data = {'TV': i1, 'Radio': i2, 'Newspaper': i3}
        features = pd.DataFrame(data, index=[0])
        return features

    st.subheader('Enter Input ')
    df = user_input_features()

    st.subheader('Input parameters')
    st.write(df)

    expected_features = ['TV', 'Radio', 'Newspaper']
    df = df[expected_features]

    if st.button('Predict'):
        scaled_features = scaler2.transform(df)
        prediction = xgb.predict(scaled_features)
        predicted_sales = prediction[0]
        st.subheader('Prediction')
        st.write(f"The predicted sales amount is: **{predicted_sales}**")

elif project == "spam":
    html_temp_subtitle = """
        <div style="background-color:#ff6347;padding:10px;margin-bottom:20px">
        <h2 style="color:white;text-align:center;">SPAM Detection</h2>
        </div>
        """
    st.markdown(html_temp_subtitle, unsafe_allow_html=True)

    st.subheader('Enter Text')


    ps = PorterStemmer()
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)


    tfidf = pickle.load(open('vectorizer2.pkl','rb'))
    model = pickle.load(open('model3b.pkl','rb'))

    st.title("SMS Spam Classifier")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("Spam")
        elif result==0:
            st.header("Not Spam")