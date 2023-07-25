#!/usr/bin/env python
# coding: utf-8

# # Spotify Data Analysis 

# ## STEP 1- Import LIBRARIES

# In[19]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load my dataset into a dataframe

# In[2]:


df_tracks = pd.read_csv(r'C:\Users\joybose\Desktop\F1,,,,DATA\tracks.csv')
df_tracks.head()


# #  check null values

# In[3]:


pd.isnull(df_tracks).sum()


# # Check info about the tracks

# In[4]:


df_tracks.info()


# # sort the values with respect to their popularity

# In[7]:


sorted_df = df_tracks.sort_values('popularity')
sorted_df


# In[8]:


df_tracks.describe().transpose()


# In[10]:


most_popular= df_tracks.query('popularity>90',inplace=False).sort_values('popularity', ascending=False)
most_popular[:10]


# # set release date as the index and change the type as datetime
# 

# In[11]:


df_tracks.set_index('release_date', inplace = True)
df_tracks.index = pd.to_datetime(df_tracks.index)
df_tracks.head()


# # Get the artist on the 18th row 
# 

# In[13]:


df_tracks[["artists"]].iloc[18]


# # Convert duration from ms to just seconds

# In[14]:


df_tracks["duration"] = df_tracks["duration_ms"].apply(lambda x: round(x/1000))
df_tracks.drop("duration_ms", inplace=True,axis=1)


# In[15]:


df_tracks.duration.head()


# # visualiztion of Correlation

# In[21]:


corr_df = df_tracks.drop(["key","mode","explicit"],axis=1).corr(method= "pearson")
plt.figure(figsize = (14,6))
heatmap= sns.heatmap(corr_df,annot= True, fmt=".1g",vmin=-1,vmax=1, center=0, cmap="inferno",linewidths=1,linecolor="Black")
heatmap.set_title("Correlation Heatmap Between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)


# In[22]:


sample_df = df_tracks.sample(int(0.004* len(df_tracks)))


# In[23]:


print(len(sample_df))


# # plot a recreational correlation map

# In[25]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df , y= "loudness",x="energy",color="c").set(title = "Loudness vs Energy Correlation")


# In[26]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df , y= "popularity",x="acousticness",color="b").set(title = "popularity vs acousticness Correlation")


# In[28]:


df_tracks['dates'] = df_tracks.index.get_level_values('release_date')
df_tracks.dates = pd.to_datetime(df_tracks.dates)
years= df_tracks.dates.dt.year


# # plot a distribution plot

# In[29]:


get_ipython().system('pip install --user seaborn==0.11.0')


# In[30]:


sns.displot(years,discrete= True, aspect=2,height=5,kind="hist").set(title="Number of songs per year")


# In[33]:


total_dr = df_tracks.duration
fig_dims = (18,7)
fig, ax = plt.subplots(figsize = fig_dims)
fig = sns.barplot(x=years, y=total_dr , ax = ax , errwidth= False).set(title= "Year vs Duration")
plt.xticks(rotation=90)


# In[38]:


df_genre = pd.read_csv(r"C:\Users\joybose\Desktop\F1,,,,DATA\SpotifyFeatures.csv")


# In[39]:


df_genre.head()


# In[41]:


plt.title("Duration of songs in different genre")
sns.color_palette("rocket", as_cmap = True)
sns.barplot(y="genre", x="duration_ms", data=df_genre)
plt.xlabel("Duration in milli seconds")
plt.ylabel("Genres")


# In[43]:


sns.set_style(style = "darkgrid")
plt.figure(figsize = (10,5))
famous = df_genre.sort_values("popularity", ascending = False).head(20)
sns.barplot(y='genre', x='popularity',data=famous).set(title= "Top 5 genres by popularity")


# In[ ]:




