# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:27:25 2023

@author: SanthosRaj
"""

import numpy as np 
import pandas as pd
import ast
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import nltk

movies = pd.read_csv("tmdb_5000_movies.csv")
credit = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credit,on="title")

#genere
#id
#keywords
#title
#overview
#cast
#crew


movies=movies[['movie_id','title','genres',"keywords",'overview','cast','crew']]

print(movies.isnull().sum())

movies.dropna(inplace=True)

movies.duplicated().sum()

movies.iloc[0].genres



def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
      l.append(i['name'])
    return l


movies["genres"] =  movies['genres'].apply(convert)


movies["keywords"]=movies['keywords'].apply(convert)


def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
           l.append(i['name'])
           counter+=1
        else:break
    return l



movies['cast'] = movies['cast'].apply(convert3)


def fetch_dir(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director":
             l.append(i['name'])
        else:break
    return l

movies['crew'] = movies["crew"].apply(fetch_dir)


movies["overview"] = movies["overview"].apply(lambda x:x.split())


movies["genres"] = movies["genres"].apply(lambda x:[i.replace(" ","") for i in x])

movies["keywords"] = movies["keywords"].apply(lambda x:[i.replace(" ","") for i in x])

movies["cast"] = movies["cast"].apply(lambda x:[i.replace(" ","") for i in x])

movies["crew"] = movies["crew"].apply(lambda x:[i.replace(" ","") for i in x])


movies['tags'] = movies["overview"]+movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"]

new_df = movies[['movie_id','title','tags']]


#steming
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()



new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new_df['tags']=new_df['tags'].apply(stem)
#converting movies to vectors and not considering stop words

cv = CountVectorizer(max_features=5000,stop_words="english")

vectors = cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

def recommend(movie):
     movie_index = new_df[new_df['title']==movie].index[0]
     distances = similarity[movie_index]
     movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
     for i in movies_list:
         print(new_df.iloc[i[0]].title)
         print(i[0])
        
import pickle
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open("similarity.pkl",'wb'))