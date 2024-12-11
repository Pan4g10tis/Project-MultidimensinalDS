import pandas as pd
import math
from datetime import datetime
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
def filter_by_categorical_inputs(results, categorical_inputs, data):
    filtered_results = []
    for result in results:
        match = True
        for attribute, values in categorical_inputs.items():
            idx = list(data.columns).index(attribute)
            if result[idx] not in values:
                match = False
                break
        if match:
            filtered_results.append(result)

    return filtered_results


def lsh_query(words, N, filtered_results, data):
    # Extract reviews and full data from the filtered results
    reviews_to_hash = [res[list(data.columns).index('review')] for res in filtered_results]
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


def kdtree_main():
    # File reading and Formating
    data = pd.read_csv("simplified_coffee.csv")

    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    columns_for_splitting = ['100g_USD', 'rating', 'review_date']
    columns_for_full_data = data.columns  # Use all columns for the final nodes

    # Create a list of tuples for KD-tree construction (only the selected columns for splitting)
    points = list(data[columns_for_splitting].to_records(index=False))
    full_data = data[columns_for_full_data].values.tolist()  # Get all the data rows

    kd_tree = build_kd_tree(points, full_data)

    # Getting Attributes From User
    available_columns = [col for col in data.columns if col != "name" and col != "review"]

    print("Available attributes for indexing:")
    for i, col in enumerate(available_columns):
        print(f"{i + 1}. {col}")

    while True:
        try:
            # User input: indices of attributes to select
            selected_indices = input("Enter the numbers of up to 4 attributes you want to index, "
                                     "separated by commas (e.g., 1,3,5,7): ")
            selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]

            # Validate selection: check length and bounds
            if len(selected_indices) > 4:
                raise ValueError("You can select at most 4 attributes.")
            if not all(0 <= idx < len(available_columns) for idx in selected_indices):
                raise ValueError("One or more selected indices are out of range.")

            # Map indices to attribute names
            selected_attributes = [available_columns[idx] for idx in selected_indices]
            print(f"You have selected the following attributes for indexing: {selected_attributes}")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

    # Define the type of each attribute
    numeric_attributes = ['100g_USD', 'rating']
    numeric_ranges = {}

    categorical_attributes = ['roaster', 'roast', 'loc_country', 'origin']
    categorical_inputs = {}

    for attribute in numeric_attributes:
        numeric_ranges[attribute] = [None, None]

    review_date_range = [None, None]

    # Collect the data from the user
    for attribute in numeric_attributes:
        if attribute in selected_attributes:
            numeric_ranges[attribute] = get_numeric_range(attribute)

    # Handle review_date range input
    if "review_date" in selected_attributes:
        review_date_range = get_review_date_range_input()

    for attribute in categorical_attributes:
        if attribute in selected_attributes:
            categorical_inputs[attribute] = get_string_input(attribute)

    # Now you have all the values for the selected attributes, numeric ranges, and categorical inputs
    print("\nYou have entered the following information:")
    print(f"Numeric Ranges: {numeric_ranges}")
    if review_date_range:
        print(f"Review Date Range (numeric): {review_date_range}")
    print(f"Categorical Inputs: {categorical_inputs}")

    # Run the query
    range_min = [numeric_ranges['100g_USD'][0], numeric_ranges['rating'][0], review_date_range[0]]
    range_max = [numeric_ranges['100g_USD'][1], numeric_ranges['rating'][1], review_date_range[1]]

    range_min = [x if x is not None else -math.inf for x in range_min]
    range_max = [x if x is not None else math.inf for x in range_max]

    results = range_query(kd_tree, range_min, range_max)

    if categorical_inputs:
        filtered_results = filter_by_categorical_inputs(results, categorical_inputs, data)
        results_to_hash = filtered_results
        print(f"Found {len(filtered_results)} results within the specified range and categorical inputs:")
        # for result in filtered_results:
            # print(result)
    else:
        results_to_hash = results
        print(f"Found {len(results)} results within the specified range:")
        # for result in results:
            # print(result)


'''
    # Get user input for words and N
    search_words = input("Enter the words to search for in reviews (separated by spaces): ").split()
    top_n = int(input("Enter the number of top matching reviews to return: "))

    # Filter final results
    final_results = lsh_query(search_words, top_n, results_to_hash, data)
    print(f"\nTop {top_n} results containing words {search_words}:")
    for result in final_results:
        print(result)
'''

kdtree_main()
