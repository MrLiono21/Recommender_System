#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
df = pd.read_csv (r'HR_DATA.csv')

# ## DATA CLEANING

# In[12]:

# drops rows that have NULL values in the following columns
df = df[df['Position'].notna()]
df = df[df['Employee_Name'].notna()]
df = df[df['State'].notna()]
df = df[df['Department'].notna()]
df = df[df['MaritalDesc'].notna()]

# In[14]:


# converts the following columns into string format
df['Position'] = df['Position'].astype(str)
df['Employee_Name'] = df['Employee_Name'].astype(str)
df['State'] = df['State'].astype(str)
df['Department'] = df['Department'].astype(str)
df['MaritalDesc'] = df['MaritalDesc'].astype(str)

# ## FEATURE ENGINEERING

# In[29]:

df['Input'] = df['Position'].map(str) + ' ' + df['State'].map(str) + ' ' + df['Department'].map(str) + ' ' + df['MaritalDesc'].map(str)

# ## CONTENT-BASED RECOMMENDER SYSTEM

# In[32]:

metadata = df.copy()

# In[34]:

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['Input'] = metadata['Input'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['Input'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[35]:


#Array mapping from feature integer indices to feature name.
tfidf.get_feature_names()[721:731]


# In[36]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[37]:


cosine_sim.shape


# In[38]:


cosine_sim[1]


# In[39]:


#Construct a reverse map of indices and Employee's Name
indices = pd.Series(metadata.index, index=metadata['Employee_Name']).drop_duplicates()


# In[40]:


indices[:10]


# In[41]:

# Function that takes in Employee Name as input and outputs most similar Employees
def get_recommendations(Employee_Name, cosine_sim=cosine_sim):
    # Get the index of the Employee that matches the Name
    idx = indices[Employee_Name]

    # Get the pairwsie similarity scores of all Employees with that Employee
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the Employees based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar Employees
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar Employees
    return metadata['Employee_Name'].iloc[movie_indices]