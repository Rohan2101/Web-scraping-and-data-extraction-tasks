#!/usr/bin/env python
# coding: utf-8

# Task 1

# In[ ]:


import pandas as pd
import numpy as np
import pandas_read_xml as pdx
import json
from scipy import stats
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET 
import seaborn as sns
import math


# xml had 2 errors that needed to be fixed, first was that all the property tags were not enclosed in a main properties tag which was fixed by adding a properties tag to enclose them all. Also the & symbol in the address were causing an error and had to removed and replace with a and with letters. Bellow you can see the xml file being read this way to show the fixed issues.

# In[ ]:


tree = ET.parse('31339646.xml') 

root = tree.getroot() 
print(root)
print(root[5][0].text)


# In[ ]:


xm = open('31339646.xml','r').read()


# In[ ]:


js = open("31339646.json",'r').read()


# In[ ]:


person_dict = json.loads(js)


# In[ ]:


df_js = pd.DataFrame()


# In[ ]:


li1 = []
li2 = []
li3 = []
li4 = []

for i55 in range(len(person_dict)):
    li1.append(person_dict[i55]['property_id'])
    li2.append(person_dict[i55]['lat'])
    li3.append(person_dict[i55]['lng'])
    li4.append(person_dict[i55]['addr_street'])


# In[ ]:


df_js['property_id'] = li1
df_js['lat'] = li2
df_js['lng'] = li3
df_js['addr_street'] = li4


# In[ ]:


import re

xml_data = '''
<property>
  <property_id>94713</property_id>
  <lat>-37.923343</lat>
  <lng>145.050533</lng>
  <addr_street>120 Tucker Road</addr_street>
</property>
<property>
  <property_id>47698</property_id>
  <lat>-37.814819</lat>
  <lng>145.043364</lng>
  <addr_street>1/287 Barkers Road</addr_street>
</property>
'''

pattern = r"<property_id>(\d+)</property_id>"
property_ids = re.findall(pattern, xm)


# In[ ]:


pattern1 = r"<lat>([\d.-]+)<\/lat>"
lat = re.findall(pattern1, xm)


# In[ ]:


pattern2 = r"<lng>([\d.-]+)<\/lng>"
lng = re.findall(pattern2, xm)


# In[ ]:


pattern3 = r"<addr_street>(.+)<\/addr_street>"
addr_street = re.findall(pattern3, xm)


# In[ ]:


print(len(property_ids),len(lat),len(lng),len(addr_street))


# In[ ]:


df = pd.DataFrame({'property_id': property_ids,
     'lat': lat,
     'lng': lng,'addr_street': addr_street
    })


# In[ ]:


df2 = pd.DataFrame({'property_id': li1,
     'lat': li2,
     'lng': li3,'addr_street': li4
    })


# In[ ]:


df1 = pd.concat([df,df2])


# In[ ]:


df1['property_id'] = df1['property_id'].astype('int')


# In[ ]:


df1 = df1.drop_duplicates(subset='property_id', keep="first")


# In[ ]:


df_stops = pd.read_csv('stops.txt')


# In[ ]:


from math import radians, cos, sin, asin, sqrt
def dist(lat1, long1, lat2, long2):
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6378 * c
    return km


# In[ ]:


def near(lat, long):
    dis = df_stops.apply(lambda row: dist(lat, long, row['stop_lat'], row['stop_lon']),axis = 1)
    return df_stops.loc[dis.idxmin(), 'stop_id']


# In[ ]:


df1['stop_id'] = df1.apply(lambda row: near(float(row['lat']), float(row['lng'])),axis=1)


# In[ ]:


df1.reset_index(drop=True)


# In[ ]:


df2 = df1
df2 = pd.merge(df2, df_stops,on='stop_id', how='left')


# In[ ]:


df2[df2.duplicated(subset=['property_id'])] 


# In[ ]:


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians,[lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2- lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6378 * c
    return km


# In[ ]:


df2['distance_to_closest_train_station'] = [haversine(float(df2.lng[i]),float(df2.lat[i]),df2.stop_lon[i],df2.stop_lat[i]) for i in range(len(df2))]


# In[ ]:


import geopandas as gpd
import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# In[ ]:


df_p = gpd.read_file('VIC_LOCALITY_POLYGON_shp.shp')


# In[ ]:


df2['suburb'] = None
for ind,x in enumerate(df_p['geometry']):
    for x1 in range(len(df2)):
        points = Point(df2['lng'][x1],df2['lat'][x1])
        if x.contains(points) == True:
            df2['suburb'][x1] = df_p['VIC_LOCA_2'][ind]


# In[ ]:


df2['sub'] = None
for ind,x in enumerate(df_p['geometry']):
    for x1 in range(len(dff)):
        points = Point(dff['lng'][x1],dff['lat'][x1])
        if x.contains(points) == True:
            dff['sub'][x1] = df_p['VIC_LOCA_2'][ind]


# In[ ]:


points = Point(144.84145,-37.783143)


# In[ ]:


point = Point(df2['lng'][8],df2['lat'][8])
points = (123123,-132123)


# In[ ]:


df_shapes = pd.read_csv("shapes.txt")


# In[ ]:


df['VIC_LOCA_2'].unique()


# In[ ]:


dff = pd.read_csv("sample_output.csv")


# In[ ]:


d12 = dff[['property_id','lat','lng','addr_street']]


# In[ ]:


html = urlopen("http://house.speakingsame.com/profile.php?q=Sunshine&sta=vic")
bsObj = BeautifulSoup(html, "html.parser")


# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


# In[ ]:


x = re.match('<!DOCTYPE html>.',str(bsObj))


# In[ ]:


from PyPDF2 import PdfReader

reader = PdfReader("Lga_to_suburb.pdf")
page = reader.pages[0]
page1 = reader.pages[1]
page2 = reader.pages[2]
text = page.extract_text() + '\n' + page1.extract_text() + '\n' + page2.extract_text()


# In[ ]:


list_names = re.findall(r"(\w+ *\w+)\s*:\s*(\[[^\]]*\])", text1)
#list_values = re.findall(r"\['([^']+)'\]", text)

print(list_names[1][1])
#print(list_values)


# In[ ]:


text1 = re.sub('\\n','', text)


# In[ ]:


li1 = []
for ind,x1 in enumerate(list_names):
    li = (list_names[ind][0],ast.literal_eval(list_names[ind][1]))
    li1.append(li)


# In[ ]:


df2['Suburb'] = None
for x6 in range(len(df2)):
    df2['Suburb'][x6] = df2['suburb'][x6].lower()


# In[ ]:


df2['lga'] = 'NA'
for id,x3 in enumerate(df2['Suburb']):
    for x4 in li1:
        for x33 in x4[1]:
            x34 = x33.replace(' ','')
            x35 = x3.replace(' ','')
            if x35.lower() == x34.lower():
                df2['lga'][id] = x4[0]
#        if x3.capitalize() in x4[1]:
#            df2['LGA'][id] = x4[0]
#        else:
#            x55 = ' ' + x3.capitalize()
#            if x55 in x4[1]:
#                df2['LGA'][id] = x4[0]


# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


# In[ ]:


sub = df2['suburb'].unique()


# In[ ]:


links = []
for x8 in df['sub'].unique():
    if ' ' in x8:
        url = x8.replace(" ", "%20")
        ht = 'http://house.speakingsame.com/profile.php?q=' + url +'&sta=vic'
        links.append(ht)
    else:
        ht = 'http://house.speakingsame.com/profile.php?q=' + x8 +'&sta=vic'
        links.append(ht)


# In[ ]:


tex = []
for x11 in links:
    html = urlopen(x11)
    bsObj = BeautifulSoup(html, "html.parser")
    tex.append(bsObj)


# In[ ]:


tex1 = []
for x11 in links[46:]:
    html = urlopen(x11)
    bsObj = BeautifulSoup(html, "html.parser")
    tex1.append(bsObj)


# In[ ]:


#links[95:]
tex2 = []
for x11 in links[95:]:
    html = urlopen(x11)
    bsObj = BeautifulSoup(html, "html.parser")
    tex2.append(bsObj)


# In[ ]:


tex3 = []
for x11 in links[144:]:
    html = urlopen(x11)
    bsObj = BeautifulSoup(html, "html.parser")
    tex3.append(bsObj)


# In[ ]:


scraped = tex[:46] + tex1[:49] + tex2[:49] + tex3


# In[ ]:


routes=pd.read_csv("routes.txt")


# In[ ]:


names = []
house = []
units = []
Muni = []
pop = []
ausb = []
income = []
price = []
for x99 in scraped:
    name = re.findall('style="height:10px"><\/div><b><font style="font-size:14px">(.+) <nobr>Median Price<\/nobr><\/font><\/b><table border="0" cellpadding="0" cellspacing="0" class="line" style="margin',str(x99))
    x111 = re.findall('Number of houses\/units<\/b><\/td><td width="20"><\/td><td>((.+) \/ (.+))<\/td><\/tr><tr><td><b>Houses\/units sales',str(x99))
    x112 = re.findall('<tr><td><b>Municipality<\/b><\/td><td width="20"><\/td><td><a (.+)<\/a><\/td><\/tr><tr><td><b>Number',str(x99))
    x113 = re.findall('>(.+)',str(x112[0]))
    x15 = re.findall('All People<\/td><td>(.+)<\/td><td>3366613',str(x99))
    x116 = re.findall('Australian Born<\/a><\/td><td>(.+)<\/td><td>65%',str(x99))
    x124 = re.findall('Weekly income<\/a><\/td><td>(.+)<\/td><td>\$1,333',str(x99))
    x125 = re.findall(';sta=vic" style="color:black;text-decoration:none" title="(.+)\/><\/a><\/td><\/tr><tr><td>Unit',str(x99))
    if x125 == []:
        x125 = re.findall(';sta=vic" style="color:black;text-decoration:none" title="(.+)\/><\/a><\/td><\/tr><tr><td>',str(x99))
    x144 = re.findall(' click to view more">(.+) <img border',str(x125[0]))
    names.append(name[0].upper())
    house.append(float(x111[0][1]))
    units.append(float(x111[0][2]))
    Muni.append(x113[0])
    pop.append(float(x15[0]))
    ausb.append(x116[0])
    income.append(x124[0])
    price.append(x144[0])
    ausb[5] = ausb[5][0:3]  


# In[ ]:


x111 = re.findall('Number of houses\/units<\/b><\/td><td width="20"><\/td><td>((.+) \/ (.+))<\/td><\/tr><tr><td><b>Houses\/units sales',str(tex[0]))


# In[ ]:


suburbs = pd.DataFrame({'suburb': names,
     'number_of_houses': house,
     'number_of_units': units,'municipality': Muni, 'population': pop,'aus_born_perc': ausb, 'median_income': income,'median_house_price': price
    })


# In[ ]:


df2 = df2.merge(suburbs,on='suburb')


# In[ ]:


x112 = re.findall('<tr><td><b>Municipality<\/b><\/td><td width="20"><\/td><td><a (.+)<\/a><\/td><\/tr><tr><td><b>Number',str(tex[0]))
x113 = re.findall('>(.+)',str(x112[0]))


# In[ ]:


x111 = re.findall('All People<\/td><td>(.+)<\/td><td>3366613',str(tex[0]))
#All People<\/td><td>(.+)<\/td><td>3366613


# In[ ]:


x116 = re.findall('Australian Born<\/a><\/td><td>(.+)<\/td><td>65%',str(tex[0]))

#Australian Born<\/a><\/td><td>(.+)<\/td><td>65%


# In[ ]:


df2[df2['LGA'] == 'NA']['suburb'].unique()


# In[ ]:


li11 = []
for v44 in range(len(li1)):
    for x1111 in v44[1]:
        


# In[ ]:


df_trips = pd.read_csv('trips.txt')
df_stopt = pd.read_csv('stop_times.txt')
df_stops = pd.read_csv('stops.txt')
df_cal = pd.read_csv('calendar.txt')
df_routes = pd.read_csv('routes.txt')


# In[ ]:


dft0 = df_trips[df_trips['service_id'] == 'T0']


# In[ ]:


dfnew = df_stopt[df_stopt['trip_id'].isin(dft0['trip_id'])]
dfn2 = dfnew[dfnew['stop_id'] == 19842]


# In[ ]:


df2['travel_min_to_MC'] = 'NA'
for iddd,x55 in enumerate(df2['stop_id']): 
    dfn1 = dfnew[dfnew['stop_id'] == x55]
    dfn3 = dfn1[dfn1['trip_id'].isin(dfn2['trip_id'])]
    dfn3['departure_time2'] = 'NA'
    for x99 in range(len(dfn3)):
        d = dfn3['departure_time'].iloc[x99].split(':')
        dfn3['departure_time2'].iloc[x99] = (int(d[0])*60) + int(d[1])
    dfn4 = dfn3[dfn3['departure_time2'] > 420]
    dfn4 =  dfn4[dfn4['departure_time2'] < 540]
    dfn5 = dfn2[dfn2['trip_id'].isin(dfn4['trip_id'])]
    dfn5['departure_time2'] = 'NA'
    for x99 in range(len(dfn5)):
        d = dfn5['arrival_time'].iloc[x99].split(':')
        dfn5['departure_time2'].iloc[x99] = (int(d[0])*60) + int(d[1])
    dfn5 = dfn5[dfn5['departure_time2'] < 600]
    dfn5 = dfn5[dfn5['departure_time2'] > 420]    
    nn = 0
    n = 0
    n3 = 0
    for xo in range(len(dfn4)):
        for xi in range(len(dfn5)):
            if dfn5['trip_id'].iloc[xi] == dfn4['trip_id'].iloc[xo]:
                if (dfn5['departure_time2'].iloc[xi] - dfn4['departure_time2'].iloc[xo]) > 0:
                    x33 = dfn5['departure_time2'].iloc[xi] - dfn4['departure_time2'].iloc[xo]
                    n = n + 1
                    nn = nn + x33
    if n != 0:
        n3 = int(nn/n)
    else:
        n3 = 'no direct trip is available'
    df2['travel_min_to_MC'].iloc[iddd] = n3
                


# In[ ]:


for x444 in range(len(df2)):
    if df2['stop_id'].iloc[x444] == 19842:
        df2['travel_min_to_MC'].iloc[x444] = 0
    elif type(df2['travel_min_to_MC'].iloc[x444]) == int: 
        df2['travel_min_to_MC'].iloc[x444] = float(df2['travel_min_to_MC'].iloc[x444])


# In[ ]:


df2['direct_journey_flag'] = 'NA'
for x4444 in range(len(df2)):
    if type(df2['travel_min_to_MC'].iloc[x4444]) == str:
        df2['direct_journey_flag'].iloc[x4444] = 0
    else:
        df2['direct_journey_flag'].iloc[x4444] = 1


# In[ ]:


dfn1 = dfnew[dfnew['stop_id'] == 20021]


# In[ ]:


dfn2 = dfnew[dfnew['stop_id'] == 19842]


# In[ ]:


dfn3 = dfn1[dfn1['trip_id'].isin(dfn2['trip_id'])]


# In[ ]:


dfn3['departure_time2'] = 'NA'
for x99 in range(len(dfn3)):
    d = dfn3['departure_time'].iloc[x99].split(':')
    dfn3['departure_time2'].iloc[x99] = (int(d[0])*60) + int(d[1])
    


# In[ ]:


dfn4 = dfn3[dfn3['departure_time2'] > 420]
dfn4 =  dfn4[dfn4['departure_time2'] < 540]


# In[ ]:


dfn5 = dfn2[dfn2['trip_id'].isin(dfn4['trip_id'])]


# In[ ]:


dfn5['departure_time2'] = 'NA'
for x99 in range(len(dfn5)):
    d = dfn5['arrival_time'].iloc[x99].split(':')
    dfn5['departure_time2'].iloc[x99] = (int(d[0])*60) + int(d[1]) 


# In[ ]:


dfn5 = dfn5[dfn5['departure_time2'] < 600]
dfn5 = dfn5[dfn5['departure_time2'] > 420]


# In[ ]:


nn = 0
n = 0
n3 = 0
for xo in range(len(dfn4)):
    for xi in range(len(dfn5)):
        if dfn5['trip_id'].iloc[xi] == dfn4['trip_id'].iloc[xo]:
            if (dfn5['departure_time2'].iloc[xi] - dfn4['departure_time2'].iloc[xo]) > 0:
                x33 = dfn5['departure_time2'].iloc[xi] - dfn4['departure_time2'].iloc[xo]
                n = n + 1
                nn = nn + x33
n3 = int(nn/n)
            


# In[ ]:


df_final = df2.drop(['stop_name','stop_short_name', 'stop_lat','stop_lon','LGA'],axis=1)


# In[ ]:


df_final['closest_train_station_id'] = df_final['stop_id']
df_final = df_final.drop('stop_id',axis=1)


# In[ ]:


for xtt in range(len(df_final)):
    if type(df_final['travel_min_to_MC'].iloc[xtt]) == float:
        df_final['travel_min_to_MC'].iloc[xtt] = format(df_final['travel_min_to_MC'].iloc[xtt],'.1f')


# In[ ]:


df_final = df_final[['property_id', 'lat','lng','addr_street','suburb','number_of_houses','number_of_units','municipality','population','aus_born_perc','median_income','median_house_price','lga','closest_train_station_id','distance_to_closest_train_station','travel_min_to_MC','direct_journey_flag']]


# In[ ]:


for xiu in range(len(df_final)):
    df_final['property_id'].iloc[xiu] = np.int64(df_final['property_id'].iloc[xiu])
    df_final['lat'].iloc[xiu] = float(df_final['lat'].iloc[xiu])
    df_final['lng'].iloc[xiu] = float(df_final['lng'].iloc[xiu])
    df_final['lat'].iloc[xiu] = float(format(df_final['lat'].iloc[xiu],'.6f'))
    df_final['lng'].iloc[xiu] = float(format(df_final['lng'].iloc[xiu],'.6f'))
    


# In[ ]:


df_final.to_csv("31339646_A3_solution.csv",index=False)


# Task2 Reshaping:

# First we make a new dataframe with the 6 columns we need to use here to transform the data

# In[ ]:


df_transform = df_final[['number_of_houses', 'number_of_units', 'population', 'aus_born_perc','median_income','median_house_price']]


# We now remove the special chartacter and convert the string values to integers for normalisation and transformation

# In[ ]:


for x111 in range(len(df_transform)):
    dd = df_transform['aus_born_perc'].iloc[x111].split('%')
    df_transform['aus_born_perc'].iloc[x111] = float(dd[0])
    dd1 = df_transform['median_income'].iloc[x111].split('$')
    dd4 = dd1[1].replace(',','')
    df_transform['median_income'].iloc[x111] = float(dd4)
    dd2 = df_transform['median_house_price'].iloc[x111].split('$')
    dd3 = dd2[1].replace(',','')
    df_transform['median_house_price'].iloc[x111] = float(dd3)


# We first use Standardization, also known as z-score normalization, which involves transforming the data so that it has zero mean and unit variance. This process is applied independently to each feature (column) in the dataset expect the target variable 'median_house_price'. The uses of standardisation in a linear are explained below:
# 1. Interpretability: Standardizing the features ensures that they are on a consistent scale, allowing for easier interpretation of the model coefficients. Since the features have been transformed to have zero mean and unit variance, the coefficients represent the change in the response variable associated with a one-standard-deviation change in the corresponding feature.
# 
# 2. Avoiding bias towards high magnitude features: Without standardization, features with larger magnitudes can dominate the learning process and have a disproportionate influence on the model. By standardizing the features, we remove this bias and ensure that all features contribute equally to the model fitting process.
# 
# 3. Improving convergence: Standardizing the data can lead to faster convergence during the training process of linear models. When features are on different scales, the optimization algorithm may take longer to find the optimal solution. Standardization brings the features to a similar scale, making the optimization process more efficient and reducing the likelihood of getting stuck in local optima.
# 
# 4. Assumptions of linearity and normality: Linear regression models assume linearity between the predictors and the response variable. Standardization helps in meeting this assumption by centering the data around zero and scaling it to unit variance. Additionally, standardization can help approximate a normal distribution for the features, which is an assumption often made in linear models.

# In[ ]:


from sklearn import preprocessing


# preprocessing.StandardScaler() creates an instance of the StandardScaler class, which is used for standardizing features by removing the mean and scaling to unit variance.
# 
# .fit() fits the StandardScaler to the specified columns in the DataFrame df_transform. It calculates the mean and standard deviation of each column to be used for scaling.
# 
# std_scale is a variable that holds the fitted StandardScaler object.
# 
# df_std = std_scale.transform(df_transform[['number_of_houses', 'number_of_units', 'population', 'aus_born_perc', 'median_income']]) applies the transformation to the specified columns in df_transform. It scales the values of these columns based on the mean and standard deviation calculated during the fitting process. The resulting scaled values are stored in the variable df_std, which is an array, not a DataFrame.

# In[ ]:


std_scale = preprocessing.StandardScaler().fit(df_transform[['number_of_houses', 'number_of_units', 'population', 'aus_born_perc','median_income']])
df_std = std_scale.transform(df_transform[['number_of_houses', 'number_of_units', 'population', 'aus_born_perc','median_income']]) # an array not a df


# Here we create new columns for our scaled data in the orginal data frame transform

# In[ ]:


df_transform['number_of_housesscaled'] = df_std[:,0] #'number_of_housesscaled' is number_of_houses scaled
df_transform['number_of_unitsscaled'] = df_std[:,1] # 'number_of_unitsscaled' is number_of_units scaled
df_transform['populationscaled'] = df_std[:,2] #'populationscaled' is population scaled
df_transform['aus_born_percscaled'] = df_std[:,3] #'aus_born_percscaled' is aus_born_perc scaled
df_transform['median_incomescaled'] = df_std[:,4] #'median_incomescaled' is median_income scaled


# In[ ]:


df_transform.describe()


# The mean after applying standardisation is 0 as it should be.

# In[ ]:


print('Mean after standardisation:\n{:.2f}, {:.2f},{:.2f},{:.2f},{:.2f}'
      .format(df_std[:,0].mean(), df_std[:,1].mean(),df_std[:,2].mean(),df_std[:,3].mean(),df_std[:,4].mean()))
print('\nStandard deviation after standardisation:\n{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'
      .format(df_std[:,0].std(), df_std[:,1].std(),df_std[:,2].std(),df_std[:,3].std(),df_std[:,4].std()))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Here with the plot below we can see how that the values of the different variables are far way and not on the same scale so its hard dont intersect.

# In[ ]:


df_transform["number_of_houses"].plot(), df_transform["number_of_units"].plot(), df_transform["population"].plot(), df_transform["aus_born_perc"].plot(), df_transform["median_income"].plot() 


# We can see Here how z-score normalization has improved the scaling of all the variable as they are now on the same scale which will make it better for a linear model. It has improved the Comparability and Interpretability of the data. The comparability is improved as standardizing the variables brings them to the same scale, making them directly comparable. When variables have different scales, their magnitudes can affect the coefficients and make it challenging to compare the relative importance of different features. Standardization resolves this issue, allowing for a fairer comparison of the effects of different variables on the target variable. The Interpretability is improved as the  the coefficients in the linear model become more interpretable. A coefficient represents the change in the target variable associated with a one-unit change in the corresponding standardized feature. Since all features now have the same scale, the coefficients become directly comparable and can be interpreted as the impact of a standardized feature on the target variable.

# In[ ]:


df_transform["number_of_housesscaled"].plot(), df_transform["number_of_unitsscaled"].plot(), df_transform["populationscaled"].plot(), df_transform["aus_born_percscaled"].plot(), df_transform["median_incomescaled"].plot()


# Below we have ten plots showing the original values of the variable vs the target variable 'median_house_price', followed by another plot of the standardized values of the variable vs the target variable 'median_house_price' for all the dependent variable. 

# In[ ]:


df_transform["number_of_houses"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["number_of_housesscaled"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["number_of_houses"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["number_of_housesscaled"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["population"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["populationscaled"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["aus_born_perc"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["aus_born_percscaled"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["median_income"].plot(), df_transform["median_house_price"].plot()


# In[ ]:


df_transform["median_incomescaled"].plot(), df_transform["median_house_price"].plot()


# With the plots shown above we can see that standardization of the dependent variables doesn't have much of an impact with respect to the target variable as the values standardized values are still very small compared to the orginal values, so it can be said that standardization is not very useful here. 
# Now to check the impact of standardization with respect to the dependent variables we use the steps below:

# In[ ]:


x = ['number_of_houses', 'number_of_units', 'population', 'aus_born_perc','median_income']
# making lists with the variable names to make it easier to plot the different variable before and after standardization
x1 = ['number_of_housesscaled', 'number_of_unitsscaled', 'populationscaled', 'aus_born_percscaled','median_incomescaled']


# Below is a plot of all the dependent variables against each other before standisation.

# In[ ]:


for x2 in x:
    for x5 in x:
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x2], df_transform[x5],
            color='green', label='input scale', alpha=0.5)

        plt.xlabel(x2)
        plt.ylabel(x5)
        plt.grid()
        plt.tight_layout()

plt.show()


# Below we plot all the dependent variables standardized against all the dependent variables non standardized

# In[ ]:


for x2 in x:
    for x3 in x1:
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x2], df_transform[x3],
            color='orange', label='input scale', alpha=0.5)


        plt.xlabel(x2)
        plt.ylabel(x3)
        plt.grid()
        plt.tight_layout()



plt.show()


# Below we plot all the dependent variables standardized against all the dependent variables standardized. 
# We can identify from these plots that standardisation doesn't have much of an impact on the values and so is not very useful and should not be used. 

# In[ ]:


for x2 in x1:
    for x3 in x1:
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x2], df_transform[x3],
            color='orange', label='input scale', alpha=0.5)


        plt.xlabel(x2)
        plt.ylabel(x3)
        plt.grid()
        plt.tight_layout()



plt.show()


# MinMax

# MinMax standardization is another type of data scaling technique commonly used for linear models. Unlike z-score normalization, which transforms data to have zero mean and unit variance, min-max standardization scales the data to a specific range, typically between 0 and 1. The uses of min max are as follows:
# 1. Scaling to a bounded range: Min-max standardization scales the data to a fixed range, often between 0 and 1. This can be particularly useful when the features have specific bounds or constraints, and you want to maintain the original data range within those bounds. By preserving the original data range, you retain the relative differences between data points.
# 
# 2. Interpretability: Min-max standardization preserves the relative ordering of values, making it easier to interpret the coefficients in a linear model. The coefficients can be directly interpreted as the change in the dependent variable associated with a unit change in the corresponding independent variable.
# 
# 3. Preserving skewness and outliers: Min-max standardization maintains the shape of the original data distribution, including any skewness or outliers. It simply linearly scales the values to fit within the desired range. This can be advantageous when the data distribution carries meaningful information, and you want to preserve those characteristics.
# 
# 4. Robustness to outliers: Min-max standardization is relatively robust to outliers. Although outliers can affect the scaling of the data within the specific range, they do not have a significant impact on the scaling of the remaining data. This can help reduce the influence of outliers on the linear model's coefficients.
# 
# 5. Maintaining sparsity: If your dataset contains sparse features (features with many zero values), min-max standardization can preserve the sparsity structure. It does not alter the zero values, as the scaling is performed based on the non-zero values only.
# The issue with min-max standardization is that it is sensitive to the range of the data. Outliers or extreme values outside the desired range can lead to loss of information or compression of data within a narrow range. In such cases, alternative scaling techniques, such as robust scaling methods, may be more appropriate.
# Below we perfrom the steps required for performing the min max standardization:
# preprocessing.MinMaxScaler(): This creates an instance of the MinMaxScaler class from the scikit-learn preprocessing module. The MinMaxScaler is used to perform min-max standardization on the data.
# 
# .fit(): This fits the MinMaxScaler to the specified subset of columns in the DataFrame df_transform. The scaler calculates the minimum and maximum values of each column, which will be used to perform the scaling.
# 
# minmax_scale.transform(): This applies the min-max standardization transformation to the specified columns using the previously fitted scaler. It scales the values in each column to a range between 0 and 1, based on the minimum and maximum values calculated during the fitting step.
# 

# In[ ]:


minmax_scale = preprocessing.MinMaxScaler().fit(df_transform[['number_of_houses', 'number_of_units', 'population', 'aus_born_perc','median_income']])
df_minmax = minmax_scale.transform(df_transform[['number_of_houses', 'number_of_units', 'population', 'aus_born_perc','median_income']])


# Below all the variable against each other before min max and before Standardization  , after  Standardization and after min max

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,4):
    for x34 in  range(0,4):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform[x[x34]],
            color='green', label='input scale', alpha=0.5)


        plt.scatter(df_std[:,x33], df_std[:,x34], color='red',
             label='Standardized u=0, s=1', alpha=0.3)
    
        plt.scatter(df_minmax[:,x33], df_minmax[:,x34],
            color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)


        plt.grid()
        plt.tight_layout()


plt.show()


# BELOW IS PLOT OF SHOWING THE COMPARISON BETWEEN THE STANDANDIZIED AND MIN MAX METHODS
# Here we can see that the min max method is better than z score as values are more closer to each other and clustered so are better with reduced outliers

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,4):
    for x34 in  range(0,4):
        f = plt.figure(figsize=(8,6))
        plt.scatter(df_std[:,x33], df_std[:,x34], color='red',
             label='Standardized u=0, s=1', alpha=0.3)
    
        plt.scatter(df_minmax[:,x33], df_minmax[:,x34],
            color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)


        plt.grid()
        plt.tight_layout()

plt.show()


# Scatter plot to see the difference of standardisation on median house price

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='input scale', alpha=0.5)


        plt.scatter(df_std[:,x33], df_transform["median_house_price"], color='red',
             label='Standardized u=0, s=1', alpha=0.3) 

        plt.title('Scatter plot to see the difference of standardisation on median house price')
        plt.xlabel(x[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()


plt.show()


# Scatter plot to see the difference of minmax on median house price

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        #plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
        #    color='green', label='input scale', alpha=0.5)


        #plt.scatter(df_std[:,x33], df_transform["median_house_price"], color='red',
        #     label='Standardized u=0, s=1', alpha=0.3) # can't print: μ = 0, σ = 0
    
        plt.scatter(df_minmax[:,x33], df_transform["median_house_price"],
            color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

        plt.title('Scatter plot to see the difference of minmax on median house price')
        plt.xlabel(x[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
    #f.savefig("z_min_max.pdf", bbox_inches='tight')

plt.show()


# Scatter plot to see the difference of minmax on median house price and std on median house price in one plot

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='input scale', alpha=0.5)    
        plt.scatter(df_minmax[:,x33], df_transform["median_house_price"],
            color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

        plt.title('Scatter plot to see the difference of minmax on median house price')
        plt.xlabel(x[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()

plt.show()


# So we can see that the values of min max are more stardardized are having a more of an impact on the data as compared to z score. As we have seen from the plots above that min max has more of an impact on this dataset and so it is recommended here instead of z score.

# Data Transformation

# Root transformation is a type of variable transformation commonly used to normalize the distribution of data and address issues related to heteroscedasticity, skewness, or non-normality. In the context of preparing data for a linear model, applying a root transformation to variables can be beneficial in certain cases. 
# Root transformation of variables can be beneficial here as it may:
# 1. Normalization: Root transformation helps normalize the distribution of variables. If the data is skewed or exhibits a non-normal distribution, applying a root transformation (such as square root or cube root) can help make the data more symmetrical and approximate a normal distribution. This can improve the assumptions of linearity, homoscedasticity, and normality required by linear models.
# 2. Linearizing relationships: In some cases, relationships between variables may not be linear. Applying a root transformation can help linearize these relationships, making them more amenable to linear modeling. This can lead to improved model fit and more reliable coefficient estimates.
# 3. Stabilizing variance: Heteroscedasticity, where the variability of the residuals changes across different levels of the independent variable, violates one of the assumptions of linear regression. Applying a root transformation can help stabilize the variance, reducing the impact of heteroscedasticity and improving the model's validity.
# 4. Improving interpretability: Root transformations can improve the interpretability of coefficients in linear models. For example, if you apply a square root transformation, the coefficient associated with the transformed variable represents the change in the dependent variable associated with a one-unit change in the square root of the independent variable. This can help make the interpretation of coefficients more meaningful and easier to grasp.
# 5. Mitigating the influence of outliers: Root transformations tend to compress the values of extreme outliers, reducing their influence on the model. This can be useful in situations where outliers have a disproportionate impact on the model's results.

# Scatter plot of all the variable against median_house_price.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='input scale', alpha=0.5)
        plt.title('Scatter plot to see the all variables vs median house price')
        plt.xlabel(x[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()


plt.show()


# In the code below we are transformating the data in the following steps:
# 1. Creating new columns: Several new columns are added to the df_transform DataFrame, each with a suffix "rt" (indicating the square root transformation). These columns are initially set to None.
# 
# 2. Looping through the DataFrame: The code iterates through each row of the df_transform DataFrame using the iterrows() method.
# 
# 3. Applying square root transformation: Within the loop, the square root transformation is applied to each value in the specified columns using the math.sqrt() function. The transformed values are then assigned to the corresponding "rt" columns in the df_transform DataFrame using the at method.
# 
# 4. Updating the index: The index variable i is incremented after each iteration to keep track of the row index.
# 
# 5. Viewing the updated DataFrame: After the loop finishes, the head() method is used to display the first few rows of the updated df_transform DataFrame, showing the newly added columns with square root transformed values.

# In[ ]:


df_transform['number_of_housesrt'] = None
df_transform['number_of_unitsrt'] = None
df_transform['populationrt'] = None
df_transform['aus_born_percrt'] = None
df_transform['median_incomert'] = None
i = 0
for row in df_transform.iterrows():
    df_transform['number_of_housesrt'].at[i] = math.sqrt(df_transform["number_of_houses"][i])
    df_transform['number_of_unitsrt'].at[i] = math.sqrt(df_transform["number_of_units"][i])
    df_transform['populationrt'].at[i] = math.sqrt(df_transform["population"][i])
    df_transform['aus_born_percrt'].at[i] = math.sqrt(df_transform["aus_born_perc"][i])
    df_transform['median_incomert'].at[i] = math.sqrt(df_transform["median_income"][i])
    
    i += 1


# In[ ]:


xyy = ['number_of_housesrt','number_of_unitsrt','populationrt','aus_born_percrt','median_incomert'] # intializing new list ot make the plotting easier. 


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt
for x33 in  range(0,5):
    for x44 in range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform[x[x44]], color='red')
        plt.xlabel(x[x33])
        plt.ylabel(x[x44])

        plt.grid()
        plt.tight_layout()

plt.show()


# PLot between the sqt values for all the variable after performaning the sqrt shows that the sqrt has reduced the data points being right skewed and has made made them more normally distributed. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
    for x44 in range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[xyy[x33]], df_transform[xyy[x44]], color='red')

        plt.xlabel(xyy[x33])
        plt.ylabel(xyy[x44])

        plt.grid()
        plt.tight_layout()


plt.show()


# If we plot the values for can see that the values are more more spread out then the orginal values. For median income the plot looks more like a linear line so it might be a good choice to use this method. Applying a root transformation appears to be linearizing relationships here, making them more amenable to linear modeling. This can lead to improved model fit and more reliable coefficient estimates.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='Origanal', alpha=0.5)

        plt.scatter(df_transform[xyy[x33]], df_transform["median_house_price"],
            color='blue', label='transformed', alpha=0.3)


        plt.xlabel(xyy[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()

plt.show()


# Square power transformation

# Power 2 transformations, also known as squared transformations, involve squaring the values of a variable. Applying power 2 transformations to variables can have several uses and benefits in the context of a linear model:
# 
# Nonlinear relationships: In some cases, the relationship between the independent variable and the dependent variable may not be strictly linear. Squaring the independent variable can capture quadratic or curvilinear relationships. This allows the linear model to account for nonlinear patterns in the data and potentially improve model fit.
# 
# Interaction terms: Power 2 transformations can be used to create interaction terms in a linear model. By squaring an independent variable and including it as an additional predictor alongside the original variable, you can capture the interaction effect between the two variables.
# 
# Enhancing linearity: Squaring a variable can help transform skewed or non-normal distributions into a more symmetrical shape, which may align better with the assumptions of a linear model. This can improve the linearity assumption and lead to more reliable coefficient estimates.
# 
# Capturing heteroscedasticity: Power 2 transformations can be useful in addressing heteroscedasticity, where the variability of the residuals changes across different levels of the independent variable. Squaring the independent variable can help stabilize the variance and achieve more constant error terms.
# 
# Interpretability: In some cases, the squared term itself may have a meaningful interpretation. For example, in economics, squaring a variable can capture the concept of diminishing marginal returns or the quadratic relationship between inputs and outputs.
# THe code below performs the following functions:
# Creating new columns: Several new columns are added to the df_transform DataFrame, each with a suffix "rt" (indicating the power 2 transformation). These columns are initially set to None.
# 
# Initializing the index: The variable i is initialized to 0. This variable will be used to keep track of the row index.
# 
# Looping through the DataFrame: The code iterates through each row of the df_transform DataFrame using the iterrows() method.
# 
# Applying power 2 transformation: Within the loop, the power 2 transformation (squaring) is applied to each value in the specified columns using the math.pow() function. The transformed values are then assigned to the corresponding "rt" columns in the df_transform DataFrame using the at method.
# 
# Updating the index: The index variable i is incremented after each iteration to move to the next row.

# In[ ]:


df_transform['number_of_housesrt'] = None
df_transform['number_of_unitsrt'] = None
df_transform['populationrt'] = None
df_transform['aus_born_percrt'] = None
df_transform['median_incomert'] = None
i = 0
for row in df_transform.iterrows():
    df_transform['number_of_housesrt'].at[i] = math.pow(df_transform["number_of_houses"][i],2)
    df_transform['number_of_unitsrt'].at[i] = math.pow(df_transform["number_of_units"][i],2)
    df_transform['populationrt'].at[i] = math.pow(df_transform["population"][i],2)
    df_transform['aus_born_percrt'].at[i] = math.pow(df_transform["aus_born_perc"][i],2)
    df_transform['median_incomert'].at[i] = math.pow(df_transform["median_income"][i],2)
    i += 1

    


# Plot between all the transformed variables

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
    for x44 in range(0,5):
        f = plt.figure(figsize=(8,6))


        plt.scatter(df_transform[xyy[x33]], df_transform[xyy[x44]], color='red')


        plt.xlabel(xyy[x33])
        plt.ylabel(xyy[x44])
        plt.grid()
        plt.tight_layout()

plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='Origanal', alpha=0.5)

    
        plt.scatter(df_transform[xyy[x33]], df_transform["median_house_price"],
            color='blue', label='transformed', alpha=0.3)


        plt.xlabel(xyy[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()

plt.show()


# The plots below show that power 2 transformation do not appear to have muuch of an impact on the data as compared to the sqrt method as the the data is not as normalized. 

# Log transformation
# A log transformation involves taking the logarithm of a variable before using it as an input to a linear model. This transformation can be applied to the independent variable, the dependent variable, or both, depending on the specific situation and the goals of the analysis. It is useful because of the following:
# 
# Skewed variables: If a variable is highly skewed, applying a log transformation can help reduce the skewness and make the variable more symmetrical. This can improve the linearity assumption between the independent variable and the dependent variable in the linear model.
# 
# Multiplicative relationships: In some cases, the relationship between variables may be multiplicative rather than additive. Taking the logarithm of the variables can transform the multiplicative relationship into an additive one. This can simplify the interpretation of the coefficients in the linear model.
# 
# Heteroscedasticity: Log transformations can help address heteroscedasticity, where the variability of the residuals changes across different levels of the independent variable. By compressing the scale of the variable, a log transformation can help stabilize the variance and reduce heteroscedasticity, improving the model's validity.
# 
# Interpretability: Log transformations can improve the interpretability of coefficients in the linear model. For example, if you take the logarithm of a variable, the coefficient associated with that variable represents the percentage change in the dependent variable for a one percent change in the independent variable. This can make the interpretation of coefficients more meaningful and easier to understand.
# 
# Outlier mitigation: Log transformations tend to compress the values of extreme outliers. This can help mitigate the influence of outliers on the model, as extreme values are shrunk closer to the mean.
# 

# In[ ]:


df_transform['number_of_housesrt'] = None
df_transform['number_of_unitsrt'] = None
df_transform['populationrt'] = None
df_transform['aus_born_percrt'] = None
df_transform['median_incomert'] = None
i = 0
for row in df_transform.iterrows():
    df_transform['number_of_housesrt'].at[i] = math.log(df_transform["number_of_houses"][i])
    df_transform['number_of_unitsrt'].at[i] = math.log(df_transform["number_of_units"][i])
    df_transform['populationrt'].at[i] = math.log(df_transform["population"][i])
    df_transform['aus_born_percrt'].at[i] = math.log(df_transform["aus_born_perc"][i])
    df_transform['median_incomert'].at[i] = math.log(df_transform["median_income"][i])
    i += 1

    
df_transform.head()


# In[ ]:


# for all the variables against each other
get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
    for x44 in range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[xyy[x33]], df_transform[xyy[x44]], color='red')

        plt.xlabel(xyy[x33])
        plt.ylabel(xyy[x44])
        plt.grid()
        plt.tight_layout()


plt.show()


# In[ ]:


#for all the variables against each other compared to orginal data.
get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
    for x44 in range(0,5):
        f = plt.figure(figsize=(8,6))


        plt.scatter(df_transform[xyy[x33]], df_transform[xyy[x44]], color='red',
          label='transformed',    alpha=0.3)
    
        plt.scatter(df_transform[x[x33]], df_transform[x[x44]],
            color='blue', label='original', alpha=0.3)


        plt.xlabel(x[x33])
        plt.ylabel(x[x44])
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()


plt.show()


# We can see with the plot below that log transformation has reduced the size of the data greatly. When we plot the variable against each other after applying log transformation we see that the data points become more normalised and start appearing in a more linear pattern so it has the most postive impact here as compared to other transformations before.It has also compressed the data to make it more closely related and decrease bias while making it normalized. It also tends to compress the values of extreme outliers. This can help mitigate the influence of outliers on the model, as extreme values are shrunk closer to the mean. It also reduces the skewness and makes the variables more symmetrical. This can improve the linearity assumption between the independent variable and the dependent variable in the linear model.

# In[ ]:


#all variables vs median_house_price before and after log transformation
get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,5):
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='Origanal', alpha=0.5)

    
        plt.scatter(df_transform[xyy[x33]], df_transform["median_house_price"],
            color='blue', label='transformed', alpha=0.3)


        plt.xlabel(xyy[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
    #f.savefig("z_min_max.pdf", bbox_inches='tight')

plt.show()


# Box-COX transformation
# The Box-Cox transformation is a technique used to transform variables in order to meet the assumptions of a linear model, particularly addressing issues related to non-normality and heteroscedasticity. The transformation is defined by a parameter, lambda (λ), which determines the type of transformation applied. The can be usefull because of the following:
# Normalization: The Box-Cox transformation can help normalize skewed variables by reducing skewness and making the variable distribution more symmetrical. This is particularly useful when the assumption of normality is violated in linear regression, as it allows for more accurate estimation of coefficients and better interpretation of results.
# 
# Linearizing relationships: In some cases, the relationship between the independent variables and the dependent variable may not be linear. The Box-Cox transformation can help linearize such relationships by applying appropriate power transformations. This enables the use of a linear model to capture the true underlying relationship more effectively.
# 
# Heteroscedasticity: Heteroscedasticity occurs when the variability of residuals is not constant across different levels of the independent variable. The Box-Cox transformation can help mitigate heteroscedasticity by stabilizing the variance. By transforming the variable, the transformed model may exhibit constant variance, making the model more reliable and valid.
# 
# Improving model fit: Applying the Box-Cox transformation can improve the fit of the linear model by reducing the impact of outliers and extreme values. Transforming the data can make it less sensitive to extreme observations and result in a better fit to the majority of the data points.
# 
# Interpretability: The Box-Cox transformation can enhance the interpretability of the coefficients in the linear model. Depending on the estimated lambda value, the interpretation can be in terms of percentage change, geometric mean, or other meaningful interpretations, making it easier to understand the relationship between the variables.
# 

# In[ ]:


for x989 in range(len(df_transform)):
    df_transform['number_of_houses'].iloc[x989] = int(df_transform['number_of_houses'].iloc[x989])
    df_transform['number_of_units'].iloc[x989] = int(df_transform['number_of_units'].iloc[x989])
    df_transform['population'].iloc[x989] = int(df_transform['population'].iloc[x989])
    df_transform['aus_born_perc'].iloc[x989] = int(df_transform['aus_born_perc'].iloc[x989])
    df_transform['median_income'].iloc[x989] = int(df_transform['median_income'].iloc[x989])


# In[ ]:


type(df_transform['aus_born_perc'][0])


# In[ ]:


#initail distribution before box cox
for xi in range(0,3):

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        prob = stats.probplot(df_transform[x[xi]], dist=stats.norm, plot=ax1)
        ax1.set_xlabel(x[xi])
        ax1.set_title('Probplot against normal distribution')


# In[ ]:


#After using box cox
for xi in range(0,3):

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        xt, _ = stats.boxcox(df_transform[x[xi]])
        prob = stats.probplot(xt, dist=stats.norm, plot=ax1)
        ax1.set_xlabel(x[xi])
        ax1.set_title('Probplot against normal distribution')


# In[ ]:


fitted_data, fitted_lambda = stats.boxcox(df_transform['number_of_houses'])


# In[ ]:


for x232 in x[0:3]:
    fitted_data, fitted_lambda = stats.boxcox(df_transform[x232])
    fig, ax = plt.subplots(1, 2)
 

    # fitted data (normal)
    sns.distplot(df_transform[x232], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="green", ax = ax[0])
 
    sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Normal", color ="green", ax = ax[1])
 
    # adding legends to the subplots
    plt.legend(loc = "upper right")
 
    # rescaling the subplots
    fig.set_figheight(5)
    fig.set_figwidth(10)
 
    print(f"Lambda value used for Transformation: {fitted_lambda}")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,3):
        fitted_data, fitted_lambda = stats.boxcox(df_transform[x[x33]])
        f = plt.figure(figsize=(8,6))

        plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
            color='green', label='Origanal', alpha=0.5)
    
        plt.scatter(fitted_data, df_transform["median_house_price"],
            color='blue', label='transformed', alpha=0.3)


        plt.xlabel(x[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
    #f.savefig("z_min_max.pdf", bbox_inches='tight')

plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,3):
        fitted_data, fitted_lambda = stats.boxcox(df_transform[x[x33]])
        f = plt.figure(figsize=(8,6))

        plt.scatter(fitted_data, df_transform["median_house_price"],
            color='blue', label='transformed', alpha=0.3)


        plt.xlabel(x[x33])
        plt.ylabel('median_house_price')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()

plt.show()


# In[ ]:


fittted = []
for x323 in x[0:3]:
    fitted_data, fitted_lambda = stats.boxcox(df_transform[x323])
    fittted.append(fitted_data)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for x33 in  range(0,3):
    for x44 in range(0,3):
        f = plt.figure(figsize=(8,6))

        #plt.scatter(df_transform[x[x33]], df_transform["median_house_price"],
 #           color='green', label='input scale', alpha=0.5)


        plt.scatter(fittted[x33], fittted[x44], color='red',
          label='transformed',    alpha=0.3)
    
        #plt.scatter(df_transform[x[x33]], df_transform[x[x44]],
        #    color='blue', label='original', alpha=0.3)


        plt.xlabel(x[x33])
        plt.ylabel(x[x44])
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
    #f.savefig("z_min_max.pdf", bbox_inches='tight')

plt.show()


# As we can see from the data given above has quite an impact on the data, it normalize skewed variables by reducing skewness and making the variable distribution more symmetrical. Also its appears to be helping linearize the relationships by applying appropriate power transformations. This enables the use of a linear model to capture the true underlying relationship more effectively. It also appear to be improving the fit of the linear model by reducing the impact of outliers and extreme values. Transforming the data can make it less sensitive to extreme observations and result in a better fit to the majority of the data points. We can see that from first two plot how much it improves the fitting of the linear model. So due to these reason it is recommended to use this transformation along with the others decided above. 

# In[ ]:




