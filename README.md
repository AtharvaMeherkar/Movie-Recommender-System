# Movie Recommender System (Content-Based)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/AI-NLP-blue?style=for-the-badge)
![Recommendation System](https://img.shields.io/badge/AI-Recommendation-red?style=for-the-badge)
![Fullstack AI](https://img.shields.io/badge/AI_Application-Fullstack-blueviolet?style=for-the-badge)

## Project Overview

This project is a web-based Movie Recommender System that utilizes a **content-based filtering** approach. It recommends movies to a user by finding titles that are similar in content (based on genres, keywords, and plot summaries) to a movie they like. The system is built with a Python Flask backend for the recommendation logic and an interactive HTML/CSS/JavaScript frontend for a dynamic user experience.

## Features

* **Content-Based Filtering:** Implements a core recommendation algorithm that suggests movies by analyzing their intrinsic attributes rather than user behavior.
* **Natural Language Processing (NLP):**
    * Utilizes **TF-IDF Vectorization** (`TfidfVectorizer` from `scikit-learn`) to convert textual movie descriptions and metadata into numerical feature vectors, capturing the importance of words.
    * Extracts and processes text features from movie genres, keywords, and plot overviews.
* **Similarity Calculation:** Employs **Cosine Similarity** to quantify the likeness between movie content vectors, identifying movies that are semantically close.
* **Interactive Web Interface:**
    * Features a clean, modern, and dark-themed UI reminiscent of streaming platforms.
    * Input field with **autocomplete suggestions** for movie titles, dynamically populated from the backend for ease of use.
    * Displays recommendations as attractive **movie cards** with dynamically generated placeholder posters.
    * **Clickable recommendations:** Users can click on a recommended movie card to automatically populate the input field with that movie's title and trigger a new search for similar titles, enabling interactive exploration.
    * Provides clear loading, error, and informational messages.
* **Data Preprocessing:** Handles loading a real-world movie dataset (`tmdb_5000_movies.csv`), including filling missing values and parsing complex JSON strings within the data (e.g., genres and keywords).
* **Full-Stack AI Application:** Combines a Python Flask backend (responsible for hosting the recommendation model and serving API endpoints) with a vanilla HTML/CSS/JavaScript frontend (for user interaction and displaying results).

## Technologies Used

* **Python 3.x:** Core language for the backend logic and ML operations.
* **Flask:** A lightweight Python web framework used for serving the application and exposing the recommendation API endpoints.
* **`pandas`:** Essential for efficient data loading, manipulation, and preprocessing of the movie dataset.
* **`scikit-learn`:** Provides key machine learning tools, specifically `TfidfVectorizer` for text vectorization and `cosine_similarity` for similarity calculation.
* **HTML5, CSS3, JavaScript:** Used for developing the interactive and responsive frontend user interface.
* **`fetch` API:** For asynchronous communication between the frontend and the Flask backend API.
* **Font Awesome:** Integrated for scalable vector icons.

## How to Download and Run the Project

### 1. Prerequisites

* **Python 3.x:** Ensure Python 3.x is installed on your system. Download from [python.org](https://www.python.org/downloads/).
* **`pip`:** Python's package installer.
* **Git:** Ensure Git is installed on your system. Download from [git-scm.com](https://git-scm.com/downloads/).
* **VS Code (Recommended):** For a smooth development experience.

### 2. Download the Project

1.  **Open your terminal or Git Bash.**
2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)AtharvaMeherkar/Movie-Recommender-System.git
    ```
3.  **Navigate into the project directory:**
    ```bash
    cd Movie-Recommender-System
    ```

### 3. Setup and Installation

1.  **Open the project in VS Code:**
    ```bash
    code .
    ```
2.  **Open the Integrated Terminal in VS Code** (`Ctrl + ~`).
3.  **Create and activate a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
    You should see `(venv)` at the beginning of your terminal prompt.
4.  **Install the required Python packages:**
    ```bash
    pip install Flask pandas scikit-learn
    ```
5.  **Download the Dataset:**
    * The project relies on the `tmdb_5000_movies.csv` dataset.
    * Download it from: `https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/download?datasetVersionNumber=1` (You might need a free Kaggle account to download).
    * **Extract** the `tmdb_5000_movies.csv` file from the downloaded ZIP archive.
    * **Place the `tmdb_5000_movies.csv` file** directly into the `Movie-Recommender-System` project folder (the same folder as `app.py`).

### 4. Execution

1.  **Ensure your virtual environment is active** in the VS Code terminal.
2.  **Set the Flask application environment variable:**
    ```bash
    # On Windows:
    $env:FLASK_APP = "app.py"
    # On macOS/Linux:
    export FLASK_APP=app.py
    ```
3.  **Run the Flask development server:**
    ```bash
    python -m flask run
    ```
    *(The first time you run this, the recommendation model will be built (TF-IDF matrix and Cosine Similarity matrix calculation). This process can take 1-2 minutes depending on your CPU. The frontend will show "Loading model and data..." until this is complete. Please be patient.)*
4.  **Open your web browser** and go to `http://127.0.0.1:5000` (or `http://localhost:5000`).
5.  **Interact with the Recommender:**
    * Wait for the "Loading model and data..." message to disappear.
    * Enter a movie title into the input field (e.g., `Avatar`, `Inception`, `The Dark Knight`). Autocomplete suggestions will appear as you type.
    * Click "Get Recommendations".
    * Explore recommendations by clicking on the suggested movie cards to get new recommendations based on that movie.

## Screenshots

*(After running the application, insert screenshots here of:)*
* *The main Movie Recommender interface with an input field and recommendations displayed as movie cards.*
* *A screenshot clearly showing the autocomplete suggestions in the input field.*
* *A screenshot demonstrating the interactive exploration (e.g., showing new recommendations after clicking a movie card).*
* *A screenshot showing the responsive layout on a smaller screen (e.g., mobile view with stacked movie cards).*

## What I Learned / Challenges Faced

* **Content-Based Recommendation Systems:** Gained comprehensive practical experience in building a fundamental recommendation system, understanding the underlying algorithms and their application.
* **Natural Language Processing (NLP) for Features:** Applied TF-IDF vectorization to transform unstructured text data (movie plots, genres, keywords) into numerical features suitable for machine learning models.
* **Similarity Metrics:** Implemented and understood the application of cosine similarity for quantifying the likeness between items based on their feature vectors.
* **Data Preprocessing for ML:** Handled data cleaning, managing missing values, and parsing complex data structures (JSON strings within CSV) from a real-world dataset for machine learning readiness.
* **Full-Stack AI Deployment:** Successfully integrated a Python Flask backend (for the recommendation model and API) with a dynamic HTML/CSS/JavaScript frontend for an interactive AI application.
* **User Experience for Recommendation Systems:** Focused on designing an intuitive browsing experience, including features like autocomplete and clickable recommendations for seamless exploration.
