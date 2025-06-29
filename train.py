import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
import joblib

nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# 1. Load Data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the datasets on 'title'
movies = movies.merge(credits, on='title')

# 2. Keep only useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# 3. Drop nulls
movies.dropna(inplace=True)


# 4. Parsing JSON-like strings to Python objects
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# Keep top 3 cast members
def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)][:3]


movies['cast'] = movies['cast'].apply(convert_cast)


# Extract director from crew
def get_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []


movies['crew'] = movies['crew'].apply(get_director)

# 5. Text Normalization
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all into one "tags" feature
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# 6. Stemming
ps = PorterStemmer()


def stem(text):
    return " ".join([ps.stem(word) for word in text.split() if word not in stopwords.words('english')])


new_df['tags'] = new_df['tags'].apply(stem)

# 7. Vectorization using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# 8. Similarity Matrix
similarity = cosine_similarity(vectors)


# 9. Recommend Function
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return f"Movie '{movie}' not found in dataset."
    idx = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        recommendations.append(new_df.iloc[i[0]].title)
    return recommendations


# 10. Save Model & Data
joblib.dump(cv, 'vectorizer.pkl')
joblib.dump(similarity, 'similarity.pkl')
new_df.to_csv('movies_cleaned.csv', index=False)

# Optional: Save full recommend system
with open('movie_recommender.pkl', 'wb') as f:
    pickle.dump(recommend, f)

print("✅ Model and vectorizer saved. Use 'recommend(movie_name)' to get suggestions.")
