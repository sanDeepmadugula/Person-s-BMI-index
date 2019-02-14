#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
os.chdir('D:\\python using jupyter\\general')


# In[3]:


data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')


# In[4]:


data.columns


# In[5]:


data.head()


# In[6]:


data.corr()


# In[7]:


f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt='0.3f',ax=ax)
plt.show()


# In[11]:


# male and female height weight ploting  for information BMI 
male = data[data.Gender=='Male']
female = data[data.Gender=='Female']
plt.plot(male.Height,color='red',label='Male Height',alpha=0.6)
plt.plot(male.Weight,color='blue',label='Male Weight')
plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('weight-height')
plt.title('Male plot')
plt.show()

plt.plot(female.Height,color="red",label= " Female Height",alpha=0.6)
plt.plot(female.Weight,color="blue",label= "Female Weight")
plt.legend(loc='lower right')
plt.xlabel("sample")
plt.ylabel("Weight-Height")
plt.title('Female Plot') 
plt.show()


# In[12]:


# male and female height weight ploting  for information BMI 
male.index = np.arange(len(male))        #index numbers ranking
female.index = np.arange(len(female))    ##index numbers ranking
plt.plot(male.Height,color="red",label= " Female Height",alpha=0.6)
plt.plot(male.Weight,color="blue",label= "Female Weight")
plt.legend(loc='upper right')
plt.title("Male")
plt.show()
plt.plot(female.Height,color="red",label= " Female Height",alpha=0.6)
plt.plot(female.Weight,color="blue",label= "Female Weight")
plt.legend(loc='upper right')
plt.title("Female")
plt.show()


# In[14]:


# Difference between height and weight for BMI header
fark = np.zeros((500))
for i in range(0,500):
    fark[i] = data['Height'][i] - data['Weight'][i]
a = np.arange(0,500,1)

plt.plot(a,fark)
plt.title('Height Weight difference')
plt.show()


# In[17]:


# Ploting for relationship between Height and Weight 
data.Height.plot(kind = 'line', color = 'g',label = 'Height',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
data.Weight.plot(color = 'r',label = 'Weight',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Sample axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Male-Female Line Plot')            # title = title of plot
plt.show()


# In[19]:


#Scatter plot for Weight with index ( for BMI descriptions)
data.plot(kind='scatter',x='Index',y='Weight',alpha=0.6,color='red')
plt.xlabel('Index')
plt.ylabel('Weight')
plt.title('Weight Index scatter plot')


# In[20]:


#filter is BMI(Body mass index) wiht Index columns
indx = male['Index'] > 4
male[indx]
indx1 = female['Index'] > 4
female[indx1]
print(len(male[indx]) - len(female[indx1]))
plt.subplot(2,1,1)
plt.plot(female[indx1]['Weight'])
plt.show()
plt.subplot(2,1,2)
plt.plot(male[indx]['Weight'])
plt.title('Male Filter-Index Weight')
plt.show()


# In[21]:


data.Height.plot(kind='hist',bins=50,figsize=(7,7))
plt.show()


# In[23]:


data.Index.plot(kind='hist',bins=50)
plt.show()


# In[24]:


series = data['Height']
print(type(series))
data_frame = data[['Height']]
print(type(data_frame))


# In[25]:


xheight = data['Height'] > 195
data[xheight]


# In[26]:


data[np.logical_and(data['Height']>195,data['Weight']>120)]


# In[27]:


for index,value in data[['Height']][0:5].iterrows():
    print(index,":",value)


# In[29]:


# The Body Mass Index (BMI) Calculator 
data["status"] = ["Obese Class II" if i == 5 else "Obese Class I" if i == 4 else "Overweight" 
                  if i == 3 else "Normal" if i == 2 else "Normal" if i == 1 else "Extremely" for i in data.Index]
data.loc[0:10,["Height","Weight","Index","status"]]


# In[31]:


# Meter to cm
dataheight = [i/100 for i in data.Height]

#data["Kg/m2"] = [round(j / dataheight[i]**2, 2) for i,j in data[["Weight"]].iterrows()]
databmı = np.zeros(500)  # array of range 500 for body mass index
i = 0
while i<len(databmı):
    databmı[i] = round( (data["Weight"][i] / (dataheight[i]**2)) , 2 )   # print(round(x,2)) example : x=  1.34554 after round(x,2) of 1,34.
    i = i + 1
data["Kg/m2"] = databmı
data.loc[:10,["Height","Weight","status","Kg/m2"]]


# In[32]:


#plot BMI index
data["Kg/m2"].plot(kind = 'line', color = 'g',label = 'Kg/m2',linewidth=1,alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc = "upper right")
plt.ylabel("Kg/m2")
plt.title("BMI(Body Mass Index) İnformatiın")
plt.show()


# In[33]:


# male and female status for bmi
male = data[data.Gender=='Male']
male.index = np.arange(len(male))

female = data[data.Gender=='Female']
female.index = np.arange(len(female))

malestatus = male.status
malestatus.index = np.arange(len(male))

femalestatus = female.status
femalestatus.index = np.arange(len(female))

print(malestatus.loc[:4])
print(femalestatus.loc[:4])


# In[34]:


#Ploting male and female kg/m2 for bmı
male['Kg/m2'].plot(kind='line',color='g',alpha=0.6,label='Male-kg/m2')
female['Kg/m2'].plot(kind='line',color='black',alpha=1,label='Female-Kg/m2')
plt.legend(loc='upper right')
plt.show()


# In[36]:


print(male['Kg/m2'].describe())
print('----------------------')
print(female['Kg/m2'].describe())


# In[37]:


#how many obese class II range in male and female
malerange = male.status == 'Obese Class II'
maleclass = male.status[malerange]
maleclass.index = np.arange(len(maleclass))
print("Max range Male human %d"%len(male))                   # %d integer numeric for %len(male) is numeric.
print("Male Status range Obese Class II :",len(maleclass))

print('--------------------------------------------------')

femalerange = female.status == 'Obese Class II'
femaleclass = female.status[femalerange]
femaleclass.index = np.arange(len(femaleclass))
print("Max range Male human %d"%len(male))                   # %d integer numeric for %len(male) is numeric.
print("Male Status range Obese Class II :",len(femaleclass))


# In[38]:



print(data['status'].value_counts(dropna=False))


# In[39]:


data.describe()


# In[40]:


data.boxplot(column='Kg/m2',by='status',figsize=(20,5))
plt.show()


# In[41]:


data_new = data.head()
data_new


# In[42]:


melted = pd.melt(frame=data_new,id_vars='Weight',value_vars=['Kg/m2'])
melted


# In[43]:


melted.pivot(index='Weight',columns='variable',values='value')


# In[44]:


# Firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row


# In[45]:


data1 = data['Kg/m2'].head()
data2= data['status'].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col


# In[46]:


data['status'] = data['status'].astype('category')
data['Index'] = data['Index'].astype('float')


# In[47]:


data1 = data.loc[:,['Height','Weight','Kg/m2']]
data1.plot()


# In[48]:


# subplots
data1.plot(subplots = True,figsize=(5,5))
plt.show()


# In[50]:


data1.plot(kind='scatter',x='Weight',y='Kg/m2')
plt.show()


# In[51]:


# hist plot  
data1.plot(kind = "hist",y = "Kg/m2",bins = 50,range= (0,80),normed = True)
plt.show()


# In[52]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Kg/m2",bins = 50,range= (0,80),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Kg/m2",bins = 50,range= (0,80),ax = axes[1],cumulative = True)
#plt.savefig('graph.png')
plt.show()


# In[53]:


# datetime = object
#parse_dates(boolean): Transform date to ISO 8601 (yyyy-mm-dd hh:mm:ss ) format
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[55]:


import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of bmı data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2['date'] = datetime_object
data2 = data2.set_index('date')
data2


# In[56]:


# Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# In[57]:


data2.resample("A").mean()


# In[58]:


data2.resample("M").mean()


# In[59]:



# We can interpolete from first value

data2.resample('M').first().interpolate('linear')


# In[60]:


data2.resample("M").mean().interpolate("linear")


# In[62]:


data.loc[1:10,"Index":'Kg/m2']


# In[63]:


# Reverse slicing 
data.loc[10:1:-1,"Index":"Kg/m2"]


# In[64]:


data.loc[1:10,'Index':]


# In[65]:


def div(n):
    return n/2
data.Index.apply(div)


# In[66]:


# Or we can use lambda function
data['Kg/m2'].apply(lambda n :n/2)


# In[68]:


print(data.index.name)
data.index.name = 'index_name'
data.head()


# In[69]:


# pivoting: reshape tool
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[70]:


# pivoting
df.pivot(index="treatment",columns = "gender",values="response")


# In[71]:


df1 = df.set_index(["treatment","gender"])
df1
# lets unstack it


# In[72]:


df1.unstack(level=0)


# In[73]:


df1.unstack(level=1)


# In[74]:


df2 = df1.swaplevel(0,1)
df2


# In[75]:


pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# In[77]:


male.groupby('status').mean()


# In[79]:


female.groupby('status').mean()


# In[80]:


male.groupby("status")[["Kg/m2"]].mean() 


# In[81]:


female.groupby("status")[["Kg/m2"]].mean() 


# In[ ]:




