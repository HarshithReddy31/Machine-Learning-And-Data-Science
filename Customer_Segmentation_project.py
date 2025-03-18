#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("Mall_Customers.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df.drop(["CustomerID"], axis = 1, inplace=True)


# In[8]:


df.head()


# In[9]:


plt.figure(figsize =(5,5))
sns.countplot(data =df , x= 'Gender')
plt.show()


# In[10]:


age18_25=df.Age[(df.Age>=18) & (df.Age<=25)]
age26_35=df.Age[(df.Age>=18) & (df.Age<=25)]
age36_45=df.Age[(df.Age>=18) & (df.Age<=25)]
age46_55=df.Age[(df.Age>=18) & (df.Age<=25)]
age56andabove=df.Age[(df.Age>=56)]
x=["18-25","26-35","36-45","46-55","55+"]
y=[len((age18_25).values),len((age26_35).values),len((age36_45).values),len((age46_55).values),len((age56andabove).values)]


# In[11]:


plt.figure(figsize=(10,8))
sns.barplot(x=x,y=y,palette="rocket")
plt.title("No of customers and ages")
plt.xlabel("Age")
plt.ylabel("No of customers")
plt.show()


# In[16]:


ai0_30=df["Annual Income (k$)"][(df["Annual Income (k$)"]>=0) &(df["Annual Income (k$)"]<=30)]
ai31_60=df["Annual Income (k$)"][(df["Annual Income (k$)"]>=31) &(df["Annual Income (k$)"]<=60)]
ai61_90=df["Annual Income (k$)"][(df["Annual Income (k$)"]>=61) &(df["Annual Income (k$)"]<=90)]
ai91_120=df["Annual Income (k$)"][(df["Annual Income (k$)"]>=91) &(df["Annual Income (k$)"]<=120)]
ai121_150=df["Annual Income (k$)"][(df["Annual Income (k$)"]>=121) &(df["Annual Income (k$)"]<=150)]


aix =["$ 0-30,000","$ 30,001-60,000","$ 60,001-90,000","$ 90,001-120,000","$ 120,001-150,000"]
aiy =[len((ai0_30).values),len((ai31_60).values),len((ai61_90).values),len((ai91_120).values),len((ai121_150).values)]


# In[18]:


plt.figure(figsize=(10,8))
sns.barplot(x=aix ,y=aiy,palette="Spectral")
plt.title("No of customers and Annual Incomes")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customer")
plt.show()


# In[19]:


ss1_20 =df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=1) & (df["Spending Score (1-100)"] <=20)]
ss21_40 =df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=21) & (df["Spending Score (1-100)"] <=40)]
ss41_60 =df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=41) & (df["Spending Score (1-100)"] <=60)]
ss61_80 =df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=61) & (df["Spending Score (1-100)"] <=80)]
ss81_100 =df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=81) & (df["Spending Score (1-100)"] <=100)]

ssx =["1-20","21-40","41-60","61-80","81-100"]
ssy = [len(ss1_20.values),len(ss21_40.values),len(ss41_60.values),len(ss61_80.values),len(ss81_100.values)]


# In[20]:



plt.figure(figsize=(15,6))
sns.barplot(x=ssx,y=ssy,palette="nipy_spectral_r")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()


# In[21]:


plt.figure(1,figsize=(15,6))
n=0
for i in ["Age","Annual Income (k$)","Spending Score (1-100)"]:
    n+=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.distplot(df[i],bins=20)
    plt.show()
    


# In[23]:


fig, ax=plt.subplots()
fig.set_size_inches(11.7,8.27)
sns.boxplot(data=df,orient="h",palette="Set2",ax=ax)


# In[25]:


plt.figure(1,figsize=(15,7))
n=0
for cols in ["Age","Annual Income (k$)","Spending Score (1-100)"]:
    n+=1
    plt.subplot(1,3,n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace=0.5, wspace =0.5)
    sns.violinplot(x=cols,y="Gender",data=df)
    plt.ylabel('Gender' if n==1 else '')
    plt.title("violin plot")
plt.show()


# In[26]:


x1=df.loc[:,["Annual Income (k$)","Spending Score (1-100)"]].values


# In[43]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(x1)
    wcss.append(km.inertia_)


# In[44]:


plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2,color="red",marker="8")
plt.xlabel("k value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()


# In[45]:


km1=KMeans(n_clusters=5)
km1.fit(x1)
y=km1.predict(x1)
df["label"]=y
df.head()


# In[46]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',hue="label",
               palette=['green','orange','brown','dodgerblue','red'],legend='full',data=df,s=60)
plt.xlabel('Annual Income(k$)')
plt.ylabel('spending score (1-100)')
plt.title('spending score(1-100) vs Annual Income (k$)')
plt.show()


# In[48]:


x4=df.loc[:,["Age","Annual Income (k$)","Spending Score (1-100)"]].values
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(x4)
    wcss.append(km.inertia_)
    
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2,color="red",marker="8")
plt.xlabel("k value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()


# In[50]:


km = KMeans(n_clusters=5)
y2= km.fit_predict(df.iloc[:,1:])
df["label"]= y2
df.head()


# In[52]:



fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(df.Age[df.label ==0],df["Annual Income (k$)"][df.label ==0],df["Spending Score (1-100)"][df.label==0], c='purple',s=60)
ax.scatter(df.Age[df.label ==1],df["Annual Income (k$)"][df.label ==1],df["Spending Score (1-100)"][df.label==1], c='red',s=60)
ax.scatter(df.Age[df.label ==2],df["Annual Income (k$)"][df.label ==2],df["Spending Score (1-100)"][df.label==2], c='blue',s=60)
ax.scatter(df.Age[df.label ==3],df["Annual Income (k$)"][df.label ==3],df["Spending Score (1-100)"][df.label==3], c='green',s=60)
ax.scatter(df.Age[df.label ==4],df["Annual Income (k$)"][df.label ==4],df["Spending Score (1-100)"][df.label==4], c='yellow',s=60)
ax.view_init(35,185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# In[ ]:




