
# %%
import numpy as np
import pandas as pd


# %%
df = pd.read_csv(r'./data/Linux_full2.csv')
df.head()


# %%
df.info()


# %%
# convert the 'Date' column to datetime format
df['DT'] = df['DT'].astype('datetime64[ns]')
df.info()


# %%
print(df.shape)


# %%
df.isnull().sum()


# %%
df.dropna(inplace=True)


# %%
import matplotlib.pyplot as plt


# %%
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(18, 4)})


# %%
df_dt = df.set_index('DT')
df_dt['Severity'].plot(linewidth=1)


# %%
marker_colors = df_dt['Severity'].replace([0, 1, 2],['g', 'r', 'b'])
plt.scatter(df_dt.index, df_dt['Severity'], c=marker_colors)
plt.show()


# %%
sns.set(rc={'figure.figsize':(5, 4)})
sns.countplot(data=df, x='Severity')


# %%
import re

def clean_content(message):

    # Converting news content to lowercase
    message = message.lower()
    # Removing characters apart from alphabets
    message = re.sub('([^A-Za-z ])+', ' ', message)
    # Removing words less than 3 characters
    message = re.sub(r'(\b.{1,2}\s)+', ' ', message)
    
    return message


# %%
df['Message'] = df['Message'].apply(clean_content)
df.sample(10)


# %%
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %%
from typing import List
def get_vectors(texts) -> List:
    text = [t for t in texts]
    my_tfidf_vectorizer = TfidfVectorizer()
    my_tfidf_vectorizer.fit(text)

    return my_tfidf_vectorizer.transform(text).toarray()

def get_cosign_sim(received_vectors):
    return cosine_similarity(received_vectors)


# %%
vectors = get_vectors(df['Message'])
vectors


# %%
# Checking the similarities amoung vectors
similarity = get_cosign_sim(vectors)
similarity


# %%
marker_colors = df_dt['Severity'].replace([0, 1],['g', 'r'])


# %%
# use PCA to reduce dimensionality from 6 to 2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = (12, 6)

pca = PCA(n_components=3)
cos_sim_pca_3d = pca.fit_transform(similarity)
retained_variance = pca.explained_variance_ratio_
print(retained_variance)
cos_sim_pca_3d


# %%
from mpl_toolkits import mplot3d
# Creating figure
fig = plt.figure(figsize = (12, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(cos_sim_pca_3d[:, 0], cos_sim_pca_3d[:, 1], cos_sim_pca_3d[:, 2], c=marker_colors.array, s=10)
    
plt.show()

