from flask import Flask, request, render_template
import pandas as pd
import joblib
import requests

app = Flask(__name__)

movies = pd.read_csv('movies_cleaned.csv')
similarity = joblib.load('similarity.pkl')

TMDB_API_KEY = "71b1bc08a1949697543a322eb4613727"


import requests
import time

def fetch_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raises an error for bad status
            data = response.json()
            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return "https://image.tmdb.org/t/p/w500" + poster_path
            return "https://via.placeholder.com/300x450?text=No+Image"
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: Failed to fetch poster for '{title}'. Retrying...")
            time.sleep(1.5)
    print(f"❌ Could not fetch poster for '{title}' after 3 attempts.")
    return "https://via.placeholder.com/300x450?text=Error"



def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return []
    idx = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]

    seen = set()
    recommended = []
    for i in movie_list:
        title = movies.iloc[i[0]].title
        if title not in seen:
            poster = fetch_poster(title)
            recommended.append({'title': title, 'poster': poster})
            seen.add(title)
        if len(recommended) == 5:
            break
    return recommended


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    movie_name = ''
    error = ''
    if request.method == 'POST':
        movie_name = request.form.get('movie')
        recommendations = recommend(movie_name)
        if not recommendations:
            error = f"No recommendations found for '{movie_name}'"
    return render_template('index.html', movies=recommendations, error=error, searched=movie_name)


if __name__ == '__main__':
    app.run(debug=True)
