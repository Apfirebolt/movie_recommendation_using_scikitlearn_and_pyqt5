import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file
movies_df = pd.read_csv('movies.csv')


def train_model():
    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the keywords column
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['keywords'].fillna(''))

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def train_model_overview():
    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the overview column
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'].fillna(''))

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

def train_model_custom():
    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the keywords column
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['original_title'].fillna(''))

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def get_recommendations(title, cosine_sim):

    # Return the 5 top matches
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies_df['title'].iloc[movie_indices]


if __name__ == "__main__":
    # Train the model
    cosine_sim = train_model_custom()
    cosine_sim_overview = train_model_overview()
    # cosine_sim = train_model()

    # Print the accuracy of the model
    print(f"Model trained with {cosine_sim.shape[0]} movies.")

    # Get movie recommendations
    movie_title = "The Dark Knight"
    recommendations = get_recommendations(movie_title, cosine_sim)
    overview_recommendations = get_recommendations(movie_title, cosine_sim_overview)

    print(f"Recommendations for '{movie_title}':")
    print(recommendations)

    print(f"Recommendations for '{movie_title}' based on overview:")
    print(overview_recommendations)