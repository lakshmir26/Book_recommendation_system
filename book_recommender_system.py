#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import numpy as np
import pandas as pd


# In[2]:


books = pd.read_csv(r"C:\Users\laksh\Downloads\Books.csv.zip")
users = pd.read_csv(r"C:\Users\laksh\Downloads\Users.csv.zip")
ratings = pd.read_csv(r"C:\Users\laksh\Downloads\Ratings.csv.zip")


# In[3]:


books.head()


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


print(books.shape)
print(users.shape)
print(ratings.shape)


# In[7]:


books.isnull().sum()


# In[8]:


users.isnull().sum()


# In[9]:


ratings.isnull().sum()


# In[10]:


books.duplicated().sum()
users.duplicated().sum()
ratings.duplicated().sum()


# # Popularity based recommender system

# In[11]:


ratings_with_name = ratings.merge(books, on='ISBN')


# In[12]:


ratings_with_name


# In[13]:


num_rating_df =  ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df


# In[14]:


avg_rating_df =  ratings_with_name.groupby('Book-Title').agg({'Book-Rating':'mean'}).reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'},inplace=True)
avg_rating_df
    


# In[15]:


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[16]:


popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False)


# In[17]:


popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)


# In[18]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)
popular_df.merge(books,on='Book-Title')


# In[19]:


popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]


# # Collaborative filtering based recommender system

# In[20]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating']> 200
padhe_likhe_users = x[x].index


# In[21]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[22]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[23]:


famous_books


# In[24]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
final_ratings.drop_duplicates()


# In[25]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt


# In[26]:


pt.fillna(0,inplace=True)


# In[27]:


pt


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


similarity_score = cosine_similarity(pt)


# In[30]:


similarity_score.shape


# In[31]:


def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    for i in similar_items:
        print(pt.index[i[0]])


# In[34]:


recommend('Year of Wonders')


# In[ ]:




