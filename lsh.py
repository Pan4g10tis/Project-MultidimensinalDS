import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
file_path = 'simplified_coffee.csv'
data = pd.read_csv(file_path)

# Preprocess reviews
reviews = data['review'].fillna("").tolist()

# Vectorize the reviews using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', binary=True)
review_vectors = vectorizer.fit_transform(reviews)

# Set up an LSH-like structure using NearestNeighbors
n_neighbors = min(100, len(reviews))  # Limiting max neighbors for performance
nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
nn_model.fit(review_vectors)


# Function to query reviews containing given words
def lsh_query(words, N):
    """
    Find the top N reviews containing the specified words using an LSH-like mechanism.

    Parameters:
    - words: List of words to search for.
    - N: Number of reviews to return.

    Returns:
    - List of tuples containing (review_text, coffee_name, similarity_score).
    """
    query_vector = vectorizer.transform([" ".join(words)])
    distances, indices = nn_model.kneighbors(query_vector, n_neighbors=N)
    results = [(data.iloc[idx]['review'], data.iloc[idx]['name'], distances[0][i])
               for i, idx in enumerate(indices[0])]
    return results


# Example usage
if __name__ == "__main__":
    search_words = ["lemon", 'espresso']
    top_n = 3
    results = lsh_query(search_words, top_n)
    for review, name, score in results:
        print(f"Coffee Name: {name}\nReview: {review}\nSimilarity Score: {score}\n")