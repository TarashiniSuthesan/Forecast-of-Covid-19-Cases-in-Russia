#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the necessary Python libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode (connected = True)


# In[2]:


#Import the datasets
data1=pd.read_csv('covid19-data-from-john-hopkins-university/CONVENIENT_global_confirmed_cases.csv')
data2=pd.read_csv('covid19-data-from-john-hopkins-university/CONVENIENT_global_deaths.csv')


# In[3]:


#Data Pre-processing
data1.dropna(axis=0,inplace=True)
data2.dropna(axis=0,inplace=True)


# In[4]:


#Data Pre-processing
data1['Country/Region']=pd.to_datetime(data1['Country/Region'])
data2['Country/Region']=pd.to_datetime(data2['Country/Region'])


# In[48]:


import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in data1.columns.values[1:]:
    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict( width=2),
            name="Confirmed cases in " + step,
            x=data1['Country/Region'],
            y=data1[step].values    ,
            marker=dict(color=[i for i in range(len(data1.columns.values[1:]))])))


    
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to country: " + data1.columns.values[1:][i]}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=100,
    currentvalue={"prefix": "Frequency: "},
    steps=steps
)]

fig.update_layout(
         title_text="Change The Slider To Change To Different Countries",

    sliders=sliders
)

fig.show()


# In[6]:


fig2 = go.Figure()

# Add traces, one for each slider step
for step in data2.columns.values[1:]:
    fig2.add_trace(
        go.Scatter(
            visible=True,
            line=dict( width=2),
            name="Deaths in " + step,
            x=data2['Country/Region'],
                        y=data2[step].values,
            marker=dict(color=[i for i in range(len(data2.columns.values[1:]))])))
steps = []
for i in range(len(fig2.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig2.data)},
              {"title": "Slider switched to country: " + data2.columns.values[1:][i]}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    currentvalue={"prefix": "Frequency: "},
    steps=steps
)]

fig2.update_layout(
     title_text="Change The Slider To Change To Different Countries",
    sliders=sliders
)

fig2.show()


# In[5]:


world = pd.DataFrame({"Country":[],"Cases":[]})
world["Country"] = data1.iloc[:,1:].columns
cases = []
for i in world["Country"]:
    cases.append(pd.to_numeric(data1[i][1:]).sum())
world["Cases"]=cases

country_list=list(world["Country"].values)
idx = 0
for i in country_list:
    sayac = 0
    for j in i:
        if j==".":
            i = i[:sayac]
            country_list[idx]=i
        elif j=="(":
            i = i[:sayac-1]
            country_list[idx]=i
        else:
            sayac += 1
    idx += 1
world["Country"]=country_list
world = world.groupby("Country")["Cases"].sum().reset_index()
world.head()
continent = pd.read_csv ("covid19-data-from-john-hopkins-university\continents2.csv")
continent["name"]=continent["name"].str.upper()
continent["name"]


# In[16]:


world["Cases Range"]=pd.cut(world["Cases"],[-1500000,50000,200000,800000,1500000,2000000],labels=["U50K","50Kto200K","200Kto800K","800Kto1.5M","1.5M+"])
alpha =[]
for i in world["Country"].str.upper().values:
    if i == "BRUNEI":
        i="BRUNEI DARUSSALAM"
    elif  i=="US":
        i="UNITED STATES" 
    if len(continent[continent["name"]==i]["alpha-3"].values)==0:
        alpha.append(np.nan)
    else:
        alpha.append(continent[continent["name"]==i]["alpha-3"].values[0])
world["Alpha3"]=alpha

fig = px.choropleth(world.dropna(),
                   locations="Alpha3",
                   color="Cases Range",
                    projection="mercator",
                    color_discrete_sequence=["khaki","yellow","orange","red","black"])
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[18]:


count = []
for i in range(1,len(data1)):
    count.append(sum(pd.to_numeric(data1.iloc[i,1:].values)))

df = pd.DataFrame()
df["Date"] = data1["Country/Region"][1:]
df["Cases"] = count
df=df.set_index("Date")

df.Cases.plot(title="Daily COVID-19 Cases in World",marker=".",figsize=(10,5),label="daily cases")
plt.ylabel("Cases")
plt.legend()
plt.show()


# In[19]:


count = []
for i in range(1,len(data2)):
       count.append(sum(pd.to_numeric(data2.iloc[i,1:].values)))

df["Deaths"] = count
df.Deaths.plot(title="Daily COVID-19 Deaths in World",marker=".",figsize=(10,5),label="daily deaths")
plt.ylabel("Deaths")
plt.legend()
plt.show()


# In[6]:


data1.plot('Country/Region',y=['US','India','Brazil','Russia'], label=['US','India','Brazil','Russia'],figsize=(10,5),title="Daily COVID-19 Cases in Countries with Highest Record")
plt.show()


# In[7]:


data2.plot('Country/Region',y=['US','India','Brazil','Russia'],figsize=(10,5),title="Daily COVID-19 Deaths in Countries with Highest Record")
plt.show()


# In[31]:


data1_sum = data1.sum(axis = 0).reset_index()
#Remove Row which isn't a date
data1_sum = data1_sum.iloc[1: , :]
#Rename Column 
data1_sum = data1_sum.rename(columns = {0:"Cases",'index':'Country'})
#New DF for Global Sums
cases = pd.DataFrame()

for a,b in data1_sum.iterrows():
    try:
        if b['Cases'] % 1 == 0:
            cases = cases.append(b)
    except:
        pass
#Sort Cases Count Descending
cases = cases.sort_values('Cases',ascending = False)


# In[42]:


cases


# In[37]:


top20 = cases.head(20)
top20['Country'] = top20['Country'].replace(['United Kingdom.11'],'UK')
top20['Country'] = top20['Country'].replace(['France.11'],'France')
top20['Country'] = top20['Country'].replace(['Netherlands.4'],'Netherlands')
top20.plot.bar(x='Country',y='Cases',color='slateblue',title="Countries with highest number of COVID-19 Cases",figsize=(10,5),label="Cases")


# In[29]:


data2_sum = data2.sum(axis = 0).reset_index()
#Remove Row which isn't a date
data2_sum = data2_sum.iloc[1: , :]
#Rename Column 
data2_sum = data2_sum.rename(columns = {0:"Deaths",'index':'Country'})
#New DF for Global Sums
deaths = pd.DataFrame()

for a,b in data2_sum.iterrows():
    try:
        if b['Deaths'] % 1 == 0:
            deaths = deaths.append(b)
    except:
        pass
#Sort Death Count Descending
deaths = deaths.sort_values('Deaths',ascending = False)


# In[30]:


deaths


# In[49]:


top20_deaths = deaths.head(20)
top20_deaths['Country'] = top20_deaths['Country'].replace(['United Kingdom.11'],'UK')
top20_deaths['Country'] = top20_deaths['Country'].replace(['France.11'],'France')
top20_deaths.plot.bar(x='Country', y='Deaths', title="Countries with highest number of COVID-19 Deaths",figsize=(10,5),label="Deaths")


# In[54]:


metrics_data = [{'FB Prophet': 0.894, 'Linear Regression': 0.67, 'Random Forest': 0.89}, {'FB Prophet': 0.975, 'Linear Regression': 0.85, 'Random Forest': 0.98}]
 
# Creates pandas DataFrame by passing
# Lists of dictionaries and row index.
R2 = pd.DataFrame(metrics_data, index =['Confirmed Cases', 'Deaths'])


# In[55]:


R2


# In[57]:


R2 = pd.DataFrame({
    "Confirmed Cases":[0.894, 0.67, 0.89],
    "Deaths":[0.975, 0.85, 0.98]
    }, 
    index=["FB Prophet", "Linear Regression", "Random Forest"]
)

#R2.plot.bar(x='Country', y='Deaths', title="Countries with highest number of COVID-19 Deaths",figsize=(10,5),label="Deaths")
R2.plot(kind="bar")
plt.title("Coefficient of determination")
plt.xlabel("Classfiers")
plt.ylabel("R2 Score")
plt.figsize=(10,5)


# In[ ]:




