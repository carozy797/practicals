import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# reading the data
st.write(
    """
    # TV advertisement system
    The system can predict the sales  of a company based on the advertisement done by the company

"""
)

# reading csv data
advert = pd.read_csv('Advertising.csv')
print(advert)

#Nature of data displayed
st.subheader("Nature of Data")
st.write(advert[0:5])
st.markdown("the total dataset shape: " +str(advert.shape))
# st.write(advert.shape)
st.subheader('Advertising scatter plot for TV shows and the sales made')
fig, ax1 = plt.subplots()
ax1.set_title("Tv Advertisement and the sales")
ax1.set_xlabel("TvAdvertisement")
ax1.set_ylabel("Sales")
ax1.scatter (advert.TV, advert.sales)
st.pyplot(fig)
st.markdown('The scatter plot present that there is a correlation between the two variables.')
st.write("""
* the correlation between the two variables is positive
* the correlation between the two variables is seen as strong
* the correlation implies that as TV adverts increases, sales also increases

""")

st.subheader("The correlation matrix for the variables")
# FInding the correlation value
mat = advert.corr(method='kendall')
st.write(mat)

#input and output variables
X = advert.drop(columns=['sales'])
y = advert.sales
# Buiding the model
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()

model.fit(X_train, y_train)
pred_model = model.predict(X_test)

# Evaluation on model
st.subheader('Model Evaluation')
rmse = np.sqrt(mean_squared_error(pred_model, y_test))
st.write("The train model rmse: " + str(rmse))
# model_pro = model.predict_proba(X_test)

st.header("making predictions")
st.markdown("It is now time to make prediction by giving different inplut values from your company...")

st.subheader("input parameters")
st.write(X.columns)
st.markdown("It is reuired to provide different inplut values from your company based on TV, Radio and newspaper Adv")


st.sidebar.header("input parameters")
st.sidebar.markdown("Slide to show values or select file option available")

# Allowing file upload
def get_input_features():
    tv_value =  st.sidebar.slider('TV', 0, 600, 100)
    radio_value = st.sidebar.slider('Radio', 0, 600, 100)
    news_value =  st.sidebar.slider('Newspaper', 0, 600, 100)
    data = {
        'TV':tv_value,
        'radio':radio_value,
        'newspaper':news_value
    }
    features = pd.DataFrame(data, index=[0]) 
    return features
file = st.sidebar.file_uploader('Upload file', type=['csv'])
if file is not None:
    X_test_new = pd.read_csv(file)
else: 
    st.write("You didnt select file but used slider")
    X_test_new = get_input_features()

# making user prediction

y_pred = model.predict(X_test_new)

st.write(f"The train model predicted: {y_pred[0]:.3f} for your input features. That will be your **sales**")
# model_pro = model.predict_proba(X_test)