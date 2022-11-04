import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px  

df = pd.read_csv('cars_cleaned_data.csv',index_col=0)


st.set_page_config(page_title='Used cars in Egyptian Market',page_icon=':car:'
                    ,layout="centered", initial_sidebar_state='auto' )


st.sidebar.image('Photo.png',width=60)
st.sidebar.subheader('Developed by [Amr Balbaa](www.linkedin.com/in/amr-balbaa)')
st.sidebar.write('If you have any constructive feedback please reach out to this email amr.balbaa@gmail.com or click this [link](www.linkedin.com/in/amr-balbaa)')



st.title("Egyptian Market Used Cars ")

st.write('''This document sets out graphical representations of the Egyptian Market used Car’s dataset.
This dataset has been scraped from the [Hatla2e](https://eg.hatla2ee.com/en) website which is one of the most famous websites in Egypt.
This dataset contains **31,194** used car Ads, each ad. Consists of a description of the car brand, model, date of the Ad.,
 model year, traveled kilometers, color, and price. 
''')
st.image('car1.jpg')
st.write('---')


##################################################################################
##Fig1: Words

st.subheader('Most popular brands in Egyptian market')
st.write('It Seems like Hyundai and Fiat are the most famous used cars in the Egyptian market')



##Fig1: Most Popular Brands
top_brands_no = st.slider('Drag the slider to show more brands',2,20)
brand_df = pd.DataFrame(df['brand'].value_counts())
brand_df['Percentage'] = round(brand_df['brand']/31194 * 100,2)

fig1 = px.bar(brand_df['Percentage'].head(top_brands_no),color=None,title='Most popular brands in Egyptian market',
                                            template='plotly_dark',labels={"value": "Percentage %", 'index' : 'Car Brand'})

st.plotly_chart(fig1)
st.write('---')
####################################################################################
##Fig2: Words
st.subheader('Car Models Average Price')
st.write('Give it a try and select your favorite  brand, you may find a suitable model and price for you.')


selected_brand = st.selectbox('Select car model',df['brand'].unique())
models_df = pd.DataFrame(df.groupby(['brand','model']).mean()['price(Thousand)'][selected_brand])
models_df.reset_index(level=0,inplace= True)
models_df.rename(columns = {'model':'Car Model', 'price(Thousand)':'Price in Thousands'}, inplace = True)
fig2 = px.bar(models_df, x='Car Model', y='Price in Thousands',title='Car models average price',template='plotly_dark',labels={"value": "Price in Thousands", 'index' : 'Car Model'})
st.plotly_chart(fig2)
st.write('---')
#########################################################################################
#Fig3: Words
st.subheader('Check which cars you can get with your budget')

low_range = st.number_input(' From')
high_range = st.number_input('To')
st.info('Prices in Thousand Egyptian Pound')
# Fig3
price_model = pd.DataFrame(df.groupby(['brand','model']).mean()['price(Thousand)'])
price_model.reset_index(level=1,inplace=True)
price_model.reset_index(level=0,inplace=True)
table1 = price_model[(price_model['price(Thousand)']>low_range) & (price_model['price(Thousand)']<high_range)]
table1['model'] = table1['brand'] + ' ' + table1['model']
table1.drop('brand',axis=1,inplace=True)
st.dataframe(table1,width=2000)
st.write('---')
##########################################################################################
# Model Price all over 4 months from 22/6/2022 to 22/10/2022
# Fig7
st.subheader('Check how car model prices fluctuate during the duration from June to October 2022 ')

st.write('''**Note:** This is not an accurate price because it’s an average of different model year,
 it just gives you a round figure of the brand/model price. However, for more accuracy, you should select the prediction page from the sidebar.''')


show_option = st.radio('Data show options',('All model ads during the whole period','Average price per day during the whole period'))


df['date'] = df['date'].apply(lambda x : datetime.strptime(x, '%d-%m-%y'))

date_group = df.groupby(['date','brand','model']).mean()
date_group.reset_index(level=[0,1,2],inplace=True)
date_group['brand_model'] = date_group['brand'] + ' ' + date_group['model']

brand_models = date_group['brand_model'].unique()
selected_brand_model = st.selectbox('Select model',brand_models)


dg = date_group.groupby(['date','brand_model']).mean()
model_price_ts = dg.drop(columns=['year','km*1000']).reset_index(level=[0,1])
cl = model_price_ts[model_price_ts['brand_model'] == selected_brand_model]
fig7 = px.line(cl, x = 'date', y = 'price(Thousand)',title='Most popular model year').update_layout( yaxis_title="Model Price", xaxis_title="Duration from 22/6/2022 to 22/10/2022")

st.subheader('Average price for {} is {},000 L.E'.format(selected_brand_model,int(cl.groupby('brand_model').mean()['price(Thousand)'])))
  
####################################################################################################
####################################################################################################




if show_option == 'All model ads during the whole period':
    
    df1 = df.copy()
    df1['brand_model_1'] = df1['brand'] + " " + df1['model'] + '-' + df1['year'].apply(lambda x: str(x))
    df1['brand_model_2'] =  df1['brand'] + " " + df1['model']


    df1_m = df1[df1['brand_model_2']==selected_brand_model]



    outlier_h = df1_m ['price(Thousand)'].mean() + (df1_m ['price(Thousand)'].mean()*3)
    outlier_l = df1_m ['price(Thousand)'].mean() - (df1_m ['price(Thousand)'].mean()*3)

    df1_m_outlier_h  = df1_m[df1_m['price(Thousand)']>outlier_h].index
    df1_m.drop(df1_m_outlier_h,axis=0,inplace=True )

    df1_m_outlier_l  = df1_m[df1_m['price(Thousand)']<outlier_l].index
    df1_m.drop(df1_m_outlier_l,axis=0,inplace=True )


    fig8 = px.scatter(df1_m, x = 'date', y = 'price(Thousand)',hover_data = ['brand_model_1','price(Thousand)'])
    st.plotly_chart(fig8)
    

elif  show_option == 'Average price per day during the whole period':
    st.plotly_chart(fig7)



st.write('---')
###########################################################################################
# Fig4: Words
st.subheader('Proportions of cars by transmission type ')
st.write('This Pie chart showing the proportion of cars by the transmission type automatic, manual or CVT')
st.write('CVT stands for continuously variable transmission and operates in a similar fashion to a traditional automatic.')

fig4 = px.pie(df, values=df['transmission_type'].value_counts(), names=df['transmission_type'].value_counts().index,color=df['transmission_type'].value_counts().index,
       color_discrete_map={'automatic':'Orange',
                                 'manual':'Purple',
                                 'CVT':'Chartreuse'},  title='Transmission Type' ,template='plotly_dark')

st.plotly_chart(fig4)
st.write('---')
##########################################################################################
# Fig5
st.subheader('Total car prices in each year ')
st.write('This chart shows share percentage of each model year ')
st.write('Here we can observe that many people like to change the model each year, also in 2015 has alot of cars in the market')

year_df = pd.DataFrame(df['year'].value_counts().head(15))
year_df['Percentage %']=round(year_df['year']/31194 * 100,1)
year_df.reset_index(level=0,inplace=True)
year_df['index']=year_df['index'].apply(lambda x : str(x))
fig5 = px.bar(year_df,x=year_df['index'],y=['Percentage %'],title='Most popular model year',template='plotly_dark',labels={"value": "Share of Total cars in this year %", 'index' : 'Year'})
st.plotly_chart(fig5)
st.write('---')
##########################################################################################
# Fig5: Words
st.subheader('Proportions of cars in Egypt by fuel type ')
st.write('However, Egypt takes serious steps to convert petrol-run cars to natural gas and Electric cars, it seems like it may take a long time. ')
st.write('Also we can recognize that Pickup trucks, buses and microbus are minority in the used car Egyptian market ') 
fig5 = px.pie(df, values=df['fuel'].value_counts(), names=df['fuel'].value_counts().index,
       title='Fuel Type' ,template='plotly_dark')
st.plotly_chart(fig5)
st.write('---')
##########################################################################################
# Fig6: Words
st.subheader('Most Popular Car Colors ')
st.write('''Monochrome** colors (Silver, Black, and White) are the most popular colors 
and that’s aligns well with this article written by BY GO FLAT OUT 
CREW, for more details click [here](https://goflatoutph.com/2020/09/22/heres-why-most-people-prefer-white-cars/)''')

fig6 = px.pie(df,values = df['color'].value_counts().head(10),names=df['color'].value_counts().head(10).index,
                color=df['color'].value_counts().head(10).index
                ,color_discrete_map={'Silver':'Gainsboro',
                                 'Black':'Black',
                                 'White':'White',
                                 'Red':'Red',
                                 'Gray':'DimGray',
                                 'Blue':'Blue',
                                 'Dark red':'FireBrick',
                                 'Light grey':'Gainsboro',
                                 'Dark blue':'MidnightBlue',
                                 'Gold':'GoldenRod'}, title='Famous Car Colors' ,template='plotly_dark')
st.plotly_chart(fig6)       

st.write('---')
##########################################################################################
st.subheader('Thank You :pray:')

st.write('Dashboard created by [Amr Balbaa](www.linkedin.com/in/amr-balbaa)')

st.write('''I have made every effort to ensure the accuracy and reliability
 of the information on this dashboard and the machine learning model. However,
  I don’t give any kind of warranty and am not responsible for any decision taken by your side.''' )
