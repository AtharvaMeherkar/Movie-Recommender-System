# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re # For text cleaning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Global Variables for Model and Data ---
movies_df = None
tfidf_vectorizer = None
cosine_sim = None
movie_titles = [] # List of movie titles for autocomplete/dropdown

# --- Data Loading and Model Building Function ---
def load_data_and_build_model():
    global movies_df, tfidf_vectorizer, cosine_sim, movie_titles
    try:
        logger.info("Loading movie data...")
        movies_df = pd.read_csv('tmdb_5000_movies.csv')

        # --- Data Preprocessing ---
        # Select relevant columns for content-based recommendation
        features = ['genres', 'keywords', 'overview']

        # Fill NaN values with empty string to avoid errors
        for feature in features:
            movies_df[feature] = movies_df[feature].fillna('')

        # Convert list-like strings to actual lists (e.g., "[{'id': 28, 'name': 'Action'}]")
        # And extract names
        def extract_names(json_str):
            if isinstance(json_str, str) and json_str.startswith('['):
                try:
                    list_of_dicts = eval(json_str) # Using eval is risky with untrusted input, but fine for fixed dataset
                    return ' '.join([d['name'] for d in list_of_dicts])
                except (SyntaxError, TypeError):
                    return ''
            return ''

        movies_df['genres'] = movies_df['genres'].apply(extract_names)
        movies_df['keywords'] = movies_df['keywords'].apply(extract_names)

        # Combine all selected features into a single string for each movie
        movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

        # --- TF-IDF Vectorization ---
        # TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors.
        # It gives higher weight to words that are rare in the corpus but frequent in a document.
        tfidf_vectorizer = TfidfVectorizer(stop_words='english') # Remove common English stop words
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])
        logger.info("TF-IDF matrix created.")

        # --- Cosine Similarity Calculation ---
        # Cosine similarity measures the cosine of the angle between two non-zero vectors.
        # It's a common metric for text similarity. A higher score means more similarity.
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logger.info("Cosine similarity matrix calculated.")

        # Create a Series mapping movie titles to their indices
        movies_df = movies_df.reset_index() # Ensure index is a column
        # Store movie titles for the frontend
        movie_titles = sorted(movies_df['title'].tolist()) # Populate global movie_titles list

        logger.info("Movie recommender model built successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading data or building model: {e}")
        return False

# --- Flask Routes ---

@app.route('/')
def index():
    # Frontend will fetch movie_titles via a separate API call
    return render_template('index.html')

# NEW ENDPOINT: To provide the list of movie titles to the frontend for autocomplete
@app.route('/get_movie_titles', methods=['GET'])
def get_movie_titles_api():
    if not movie_titles:
        # If titles aren't loaded yet, try to load them (though they should be on app startup)
        if not load_data_and_build_model():
            return jsonify({'error': 'Failed to load movie data on demand.'}), 500
    return jsonify({'titles': movie_titles})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_title = data.get('movie_title', '').strip()

    if not movie_title:
        return jsonify({'error': 'Please provide a movie title.'}), 400

    if movies_df is None or cosine_sim is None:
        # This check ensures model is ready. If not, try to load (should be done on startup)
        if not load_data_and_build_model():
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500

    recommendations = get_recommendations(movie_title, cosine_sim, movies_df)

    if not recommendations:
        return jsonify({'message': f'No recommendations found for "{movie_title}". Please try another title or check spelling.'}), 200

    return jsonify({'recommendations': recommendations})

# --- Recommendation Function ---
def get_recommendations(movie_title, cosine_sim_matrix, df, num_recommendations=10):
    """
    Generates movie recommendations based on cosine similarity.
    """
    title_lower = movie_title.lower().strip()
    
    # Find the index of the movie that matches the title
    # Using .values for direct comparison with numpy array for efficiency
    if title_lower in df['title'].str.lower().values:
        idx = df[df['title'].str.lower() == title_lower].index[0]
    else:
        # Basic fuzzy matching for titles
        # Find titles that contain the input string
        matches = [t for t in df['title'].tolist() if title_lower in t.lower()]
        if not matches:
            return [] # No close matches found
        
        # For simplicity, pick the first close match
        logger.warning(f"Exact match not found for '{movie_title}'. Using closest match: '{matches[0]}'")
        idx = df[df['title'] == matches[0]].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (excluding itself)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top N most similar movies
    return df['title'].iloc[movie_indices].tolist()


# --- Run the Flask app ---
if __name__ == '__main__':
    # Load data and build model when the app starts
    if load_data_and_build_model():
        app.run(debug=True, port=5000)
    else:
        logger.critical("Failed to load data or build model. Exiting.")