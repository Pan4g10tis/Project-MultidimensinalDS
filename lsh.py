from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def lsh_query(words, N, filtered_results, review_index):
    """
    Perform LSH-based search for reviews containing the specified keywords.
    :param words: List of words to search for in reviews.
    :param N: Number of nearest neighbors to return.
    :param filtered_results: List of rows from the dataset (filtered by KD-Tree and categorical conditions).
    :param review_index: Index of the 'review' column in the dataset.
    :return: List of tuples containing the matching rows and their cosine distances.
    """
    if not words or N <= 0:
        return []

    # Extract reviews from the filtered results
    reviews_to_hash = [res[review_index] for res in filtered_results]

    # Vectorize the reviews
    vectorizer = CountVectorizer(stop_words='english', binary=True)
    review_vectors = vectorizer.fit_transform(reviews_to_hash)

    # Fit the NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=min(N, len(reviews_to_hash)), metric='cosine', algorithm='brute')
    nn_model.fit(review_vectors)

    # Transform the query words into a vector
    query_vector = vectorizer.transform([" ".join(words)])

    # Perform the nearest neighbors search
    distances, indices = nn_model.kneighbors(query_vector, n_neighbors=min(N, len(reviews_to_hash)))

    # Collect results with full data
    return [(filtered_results[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
