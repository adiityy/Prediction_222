import streamlit as st
st.title('😃My first app - This app contains Practice and Building app')


#the addition of sidebar changes the userphase 

st_name = st.sidebar.text_input('Enter your name', 'Hello Professor')

# approach 1. st.write(f'Hello {st_name}!')

#approach 2
st.write('Hello', st_name,'!')


#Continue from Class notebook
st.markdown("Let us create a Mathematical Application") #Markdown

st.subheader("Different Lesson will be learnt")# Subheader

st.caption("Lesson 1 - Displaying Text") #Caption

st.code("x=2021")

st.latex(r''' 
a+a r^1+a r^2+a r^3 ''')



# LESSON 2
st.caption("Lesson 2 - Input Widgets") #Caption

st.checkbox('yes')

st.button('Click')

st.radio('Pick your gender', ['Male','Female', 'Others'])

st.sidebar.selectbox('Pick your gender',['Male','Female'])

st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])

st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])

st.slider('Pick a number', 0,50)


#STILL PRACTICING
st.caption("Lesson 3 - Input Widget") #Caption

st.number_input('Pick a number', 0,10)

st.text_input('Email address')

st.date_input('Travelling date')

st.time_input('School time')

st.text_area('Description')

st.file_uploader('Upload a photo')

st.color_picker('Choose your favorite color')


# Lesson Continue
import time
st.caption("Lesson 4 - Displaying Progress Status") #Caption

st.balloons()

st.subheader("Progress bar")
st.progress(10)

st.subheader("wait for the execution")
with st.spinner('Wait for it...'): 
  time.sleep(10)

st.success("You did it !")
st.error("Error")
st.warning("Warning")
st.info("It's easy to build a streamlit app")


#MAKING GRAPHS
st.caption("Lesson 5 - Displaying Graphs") #Caption

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

rand=np.random.normal(1, 2, size=21)
fig, ax = plt.subplots()
ax.hist(rand, bins=15)#color = pink
st.pyplot(fig)


#Making Line graph
st.caption("Lesson 6 - Line graph") #Caption
import streamlit as st

import pandas as pd
import numpy as np
df= pd.DataFrame( 
    np.random.randn(10, 2), 
    columns=['x', 'y'])
st.line_chart(df)

#Lesson 7
st.caption("Lesson 7 - Bar chart") #Caption

import streamlit as st

import pandas as pd

import numpy as np
df=pd.DataFrame( 
 np.random.randn(10, 2), 
  columns=['x', 'y'])
st.bar_chart(df)


#Lesson 8
st.caption("Lesson 8 - Area Graph") #Caption

import streamlit as st
import pandas as pd
import numpy as np
df=pd.DataFrame( 
 np.random.randn(10, 2), 
 columns=['x', 'y'])
st.area_chart(df)



#Lesson 9
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.caption("Lesson 8 - Altair Chart") # Corrected caption text for consistency

df = pd.DataFrame(
    np.random.randn(500, 3),
    columns=['a', 'b', 'c'])

c = alt.Chart(df).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

st.altair_chart(c, use_container_width=True) # Changed "True" to be a boolean rather than a string


#Lesson 10
st.caption("Lesson 10 - Display Map Streamlit") # Corrected caption text for consistency


import pandas as pd
import numpy as np
import streamlit as st
df =pd.DataFrame(
   np.random.randn(500, 2) / [50, 50] + [37.76, -122.4],
  columns=['lat','lon'])
st.map(df)





#BUILDING MACHINE LEARNING APPLICATION
st.title('📠 Build a machine learning application 🖥')

st.subheader("PREDICTIVE MODEL")# Subheader

st.caption("Random Forest Classifier") #Caption

st.caption("Step 1 - Import Necessary Libraries") #Caption

import streamlit as st
import pandas as pd
import numpy as np
import pickle #to load a saved model
import base64 #to open .gif files in streamlit app

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
  feature_dict = {"No":1,"Yes":2}
  for key,value in feature_dict.items():
    if val == key:
       return value

def get_value(val,my_dict):
   for key,value in my_dict.items():
       if val == key:
           return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages



if app_mode=='Home':
  st.title('LOAN PREDICTION :')
  st.image('loan_image.jpg')
  st.markdown('Dataset :')
  data=pd.read_csv('test.csv')
  st.write(data.head())
  st.markdown('Applicant Income VS Loan Amount ')
  st.bar_chart(data[['ApplicantIncome','LoanAmount']].head(20))

elif app_mode == 'Prediction':
  st.subheader('Sir/Ma , You need to fill all necessary information in order to get a reply to your loan request !')
  st.sidebar.header("Informations about the client :")
  gender_dict = {"Male":1,"Female":2}
  feature_dict = {"No":1,"Yes":2}
  edu={'Graduate':1,'Not Graduate':2}
  prop={'Rural':1,'Urban':2,'Semiurban':3}
  ApplicantIncome=st.sidebar.slider('ApplicantIncome',0,10000,0,)
  CoapplicantIncome=st.sidebar.slider('CoapplicantIncome',0,10000,0,)
  LoanAmount=st.sidebar.slider('LoanAmount in K$',9.0,700.0,200.0)
  Loan_Amount_Term=st.sidebar.selectbox('Loan_Amount_Term',
  (12.0,36.0,60.0,84.0,120.0,180.0,40.0,300.0,360.0))
  Credit_History=st.sidebar.radio('Credit_History',(0.0,1.0))
  Gender=st.sidebar.radio('Gender',tuple(gender_dict.keys()))
  Married=st.sidebar.radio('Married',tuple(feature_dict.keys()))
  Self_Employed=st.sidebar.radio('Self Employed',tuple(feature_dict.keys()))
  Dependents=st.sidebar.radio('Dependents',options=['0','1' , '2' , '3+'])
  Education=st.sidebar.radio('Education',tuple(edu.keys()))
  Property_Area=st.sidebar.radio('Property_Area',tuple(prop.keys()))

  class_0 , class_3 , class_1,class_2 = 0,0,0,0
  if Dependents == '0':
        class_0 = 1
  elif Dependents == '1':
        class_1 = 1
  elif Dependents == '2' :
        class_2 = 1
  else: class_3= 1
  Rural,Urban,Semiurban=0,0,0
  if Property_Area == 'Urban' :
    Urban = 1
  elif Property_Area == 'Semiurban' :
    Semiurban = 1
else : Rural=1


st.radio('Pick your gender', ['Male', 'Female'])

data1={'Gender':Gender, 'Married':Married, 'Dependents':
[class_0,class_1,class_2,class_3], 'Education':Education,
'ApplicantIncome':ApplicantIncome, 'CoapplicantIncome':CoapplicantIncome, 'Self Employed':Self_Employed, 'LoanAmount':LoanAmount,
'Loan_Amount_Term':Loan_Amount_Term, 'Credit_History':Credit_History,
'Property_Area':[Rural,Urban,Semiurban], }

feature_list=[ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,get_value(Gender,gender_dict),get_fvalue(Married),data1['Dependents']
[0],data1['Dependents'][1],data1['Dependents'][2],data1['Dependents']
[3],get_value(Education,edu),get_fvalue(Self_Employed),data1['Property_Area']
[0],data1['Property_Area'][1],data1['Property_Area'][2]]
single_sample = np.array(feature_list).reshape(1,-1)



data_url = ""
data_url_no = ""


if st.button("Predict"):
          file_ = open("6m-rain.gif", "rb")
          contents = file_.read()
          data_url = base64.b64encode(contents).decode("utf-8")
          file_.close()
          
          file = open("green-cola-no.gif", "rb")
          contents = file.read()
          data_url_no = base64.b64encode(contents).decode("utf-8")
          file.close()

loaded_model = pickle.load(open('RF.sav', 'rb'))
prediction = loaded_model.predict(single_sample)
if prediction[0] == 0 :
          st.error( 'According to our Calculations, you will not get the loan from Bank' )
          st.markdown( f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">',
          unsafe_allow_html=True,)
elif prediction[0] == 1 :
          st.success( 'Congratulations!! you will get the loan from Bank' )
          st.markdown( f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
          unsafe_allow_html=True,
