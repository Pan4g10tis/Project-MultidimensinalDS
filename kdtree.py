import pandas as pd
import math
from datetime import datetime
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def convert_date_to_numeric(date_str):
    # Convert 'Month Year' date format to numeric YYYYMM format.
    try:
        return int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")


# Making The KD Tree
class KDTreeNode:
    def __init__(self, point, full_data, left=None, right=None):
        self.point = point  # A 3D point for splitting (used for tree structure)
        self.full_data = full_data  # Full row of data (all columns)
        self.left = left  # Left subtree
        self.right = right  # Right subtree


def build_kd_tree(points, full_data, depth=0):
    if not points:
        return None

    # Cycle through dimensions 0, 1, 2 for splitting
    k = 3  # The number of dimensions to use for splitting
    axis = depth % k

    # Sort points based on the current axis (this is for splitting)
    sorted_indices = sorted(range(len(points)), key=lambda i: points[i][axis])
    median = len(points) // 2

    # Create a node that holds the full row of data (all columns)
    return KDTreeNode(point=points[sorted_indices[median]],  # This is the point used for splitting
                      full_data=full_data[sorted_indices[median]],  # The full row of data (all columns)
                      left=build_kd_tree([points[i] for i in sorted_indices[:median]],
                                         [full_data[i] for i in sorted_indices[:median]], depth + 1),
                      right=build_kd_tree([points[i] for i in sorted_indices[median + 1:]],
                                          [full_data[i] for i in sorted_indices[median + 1:]], depth + 1))


# TESTING ONLY
def print_kd_tree(node, depth=0):
    if node is not None:
        print(depth, f"Node: {node.point}")
        print_kd_tree(node.left, depth + 1)
        print_kd_tree(node.right, depth + 1)


# Get range inputs for numeric attributes
def get_numeric_range(attribute_name):
    while True:
        try:
            range_input = input(f"Enter the range for {attribute_name} as 'min,max' (e.g., 1,5): ")
            min_value, max_value = [float(x.strip()) for x in range_input.split(',')]
            return min_value, max_value
        except ValueError:
            print("Invalid input. Please enter the range as two numeric values separated by a comma.")


# Get the review date input and convert it
def get_review_date_range_input():
    while True:
        try:
            date_range_input = input("Enter the review date range as 'min_date,max_date' in 'Month Year' format "
                                     "(e.g., January 2023,December 2024): ")
            min_date_str, max_date_str = [x.strip() for x in date_range_input.split(',')]
            min_date = convert_date_to_numeric(min_date_str)
            max_date = convert_date_to_numeric(max_date_str)
            return min_date, max_date
        except ValueError as e:
            print(f"Invalid date format: {e}. Please try again.")


# Get categorical string inputs for other attributes
def get_string_input(attribute_name):
    while True:
        string_input = input(f"Enter the values for {attribute_name} separated by commas (e.g., value1,value2): ").strip()
        if string_input:
            return [val.strip() for val in string_input.split(",")]
        else:
            print("Input cannot be empty. Please try again.")


# Run the range query
def range_query(node, range_min, range_max, depth=0, results=None):
    if results is None:
        results = []

    if node is None:
        return results

    k = len(range_min)  # Number of dimensions
    axis = depth % k  # Splitting axis

    # Check if the current point is within the range
    if all(range_min[dim] <= node.point[dim] <= range_max[dim] for dim in range(k)):
        results.append(node.full_data)

    # Explore the left and right children based on the splitting axis
    if node.point[axis] >= range_min[axis]:  # Potential overlap with the left subtree
        range_query(node.left, range_min, range_max, depth + 1, results)
    if node.point[axis] <= range_max[axis]:  # Potential overlap with the right subtree
        range_query(node.right, range_min, range_max, depth + 1, results)

    return results


# Filter for the categorical arguments
def filter_by_categorical_inputs(results, categorical_inputs, attribute_indices):
    """
    Filter results based on categorical conditions.
    :param results: List of rows from the dataset.
    :param categorical_inputs: Dictionary of categorical conditions (e.g., {"loc_country": ["United States"]}).
    :param attribute_indices: Dictionary mapping attribute names to column indices.
    :return: Filtered list of rows.
    """
    filtered_results = []
    for result in results:
        match = True
        for attr, values in categorical_inputs.items():
            idx = attribute_indices[attr]
            if attr in ["roaster", "roast", "loc_country", "origin"]:  # Non-numeric attributes
                # Normalize the dataset value and search values for comparison
                dataset_value = str(result[idx]).strip().lower()
                search_values = [val.strip().lower() for val in values]
                if dataset_value not in search_values:
                    match = False
                    break
        if match:
            filtered_results.append(result)
    return filtered_results


def lsh_query(words, N, filtered_results, review_index):
    # Extract reviews and full data from the filtered results
    reviews_to_hash = [res[review_index] for res in filtered_results]
    full_data = filtered_results  # Full rows of data corresponding to the reviews

    # Vectorize these reviews
    vectorizer = CountVectorizer(stop_words='english', binary=True)
    review_vectors = vectorizer.fit_transform(reviews_to_hash)

    # Fit the NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=min(N, len(reviews_to_hash)), metric='cosine', algorithm='brute')
    nn_model.fit(review_vectors)

    # Transform the query words and perform LSH search
    query_vector = vectorizer.transform([" ".join(words)])
    distances, indices = nn_model.kneighbors(query_vector, n_neighbors=min(N, len(reviews_to_hash)))

    # Collect results with full data
    return [(full_data[idx], distances[0][i]) for i, idx in enumerate(indices[0])]


def run_batch_queries(query_file, kd_tree, data):
    total_time = 0

    with open(query_file, "r") as file:
        queries = file.readlines()

    for query in queries:
        try:
            print(f"Processing query: '{query.strip()}'")
            parts = query.strip().split(',')
            if len(parts) != 6:
                print(f"Skipping invalid query (wrong number of parts): {query.strip()}")
                continue

            usd_min, usd_max = float(parts[0]), float(parts[1])
            rating_min, rating_max = float(parts[2]), float(parts[3])
            date_min, date_max = map(lambda date: convert_date_to_numeric(date.strip()), parts[4:6])

            print(f"Parsed USD range: {usd_min}, {usd_max}")
            print(f"Parsed rating range: {rating_min}, {rating_max}")
            print(f"Parsed date range: {date_min}, {date_max}")

            range_min = [usd_min, rating_min, date_min]
            range_max = [usd_max, rating_max, date_max]

            range_min = [x if x is not None else -math.inf for x in range_min]
            range_max = [x if x is not None else math.inf for x in range_max]

            start = time.time()
            results = range_query(kd_tree, range_min, range_max)
            end = time.time()

            query_time = end - start
            total_time += query_time

            print(f"Query: {query.strip()} -> Results: {len(results)} in {query_time:.4f} seconds")

        except Exception as e:
            print(f"Error processing query '{query.strip()}': {e}")

    print(f"Total time for all queries: {total_time:.4f} seconds")
    return total_time


def test_queries():
    query_file = "queries.txt"  # Path to your file
    data = pd.read_csv("simplified_coffee.csv")
    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    # Build KD-tree
    columns_for_splitting = ['100g_USD', 'rating', 'review_date']
    points = list(data[columns_for_splitting].to_records(index=False))
    full_data = data.values.tolist()
    kd_tree = build_kd_tree(points, full_data)

    # Run batch queries
    total_execution_time = run_batch_queries(query_file, kd_tree, data)
    print(f"Total execution time for all queries: {total_execution_time:.4f} seconds")


def kdtree_main(selected_attributes=None, conditions=None):
    """
    Main function for KD-tree search.
    :param selected_attributes: List of selected attributes (e.g., ["100g_USD", "rating"]).
    :param conditions: Dictionary of conditions for the selected attributes.
                      For numeric attributes, the value is a tuple (min_value, max_value).
                      For non-numeric attributes, the value is a string or list of strings.
    :return: List of matching rows.
    """
    if selected_attributes is None:
        selected_attributes = []
    if conditions is None:
        conditions = {}

    # File reading and formatting
    data = pd.read_csv("simplified_coffee.csv")
    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    # Define the type of each attribute
    numeric_attributes = ['100g_USD', 'rating', 'review_date']
    categorical_attributes = ['roaster', 'roast', 'loc_country', 'origin']

    # Extract numeric and categorical conditions
    numeric_ranges = {}
    categorical_inputs = {}

    for attr, value in conditions.items():
        if attr in numeric_attributes:
            # Numeric condition: value is a tuple (min_value, max_value)
            numeric_ranges[attr] = value
        elif attr in categorical_attributes:
            # Categorical condition: value is a string or list of strings
            if isinstance(value, str):
                categorical_inputs[attr] = [val.strip().lower() for val in value.split(",")]
            elif isinstance(value, list):
                categorical_inputs[attr] = [val.strip().lower() for val in value]

    # Build KD-tree
    columns_for_splitting = ['100g_USD', 'rating', 'review_date']
    points = list(data[columns_for_splitting].to_records(index=False))
    full_data = data.values.tolist()  # Get all the data rows
    kd_tree = build_kd_tree(points, full_data)

    # Prepare range_min and range_max for the range query
    range_min = []
    range_max = []
    for col in columns_for_splitting:
        if col in numeric_ranges:
            min_val, max_val = numeric_ranges[col]
            range_min.append(min_val if min_val is not None else -math.inf)
            range_max.append(max_val if max_val is not None else math.inf)
        else:
            range_min.append(-math.inf)
            range_max.append(math.inf)

    # Perform range query
    start = time.time()
    results = range_query(kd_tree, range_min, range_max)
    end = time.time()
    length = end - start

    # Filter results based on categorical conditions (if any)
    if categorical_inputs:
        attribute_indices = {attr: list(data.columns).index(attr) for attr in categorical_inputs}
        filtered_results = filter_by_categorical_inputs(results, categorical_inputs, attribute_indices)
        results_to_hash = filtered_results
        print(f"Found {len(filtered_results)} results within the specified range and categorical inputs.")
    else:
        results_to_hash = results
        print(f"Found {len(results)} results within the specified range.")

    print(f"Search time: {length:.4f} seconds")

    # Run test queries (unchanged)
    test_queries()

    # Return the matching rows
    return results_to_hash


'''
    # Get user input for words and N
    search_words = input("Enter the words to search for in reviews (separated by spaces): ").split()
    top_n = int(input("Enter the number of top matching reviews to return: "))

    # Filter final results
    review_index = list(data.columns).index('review')
    final_results = lsh_query(search_words, top_n, results_to_hash, review_index)
    print(f"\nTop {top_n} results containing words {search_words}:")
    for result in final_results:
        print(result)
'''
