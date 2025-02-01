import pandas as pd
import math
from datetime import datetime
from lsh import lsh_query


def convert_date_to_numeric(date_str):
    # Convert 'Month Year' date format to numeric YYYYMM format.
    try:
        return int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")


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


def kdtree_main(selected_attributes=None, conditions=None, review_keywords=None, num_neighbors=None):
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
            numeric_ranges[attr] = value
        elif attr in categorical_attributes:
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
    results = range_query(kd_tree, range_min, range_max)

    # Filter results based on categorical conditions (if any)
    if categorical_inputs:
        attribute_indices = {attr: list(data.columns).index(attr) for attr in categorical_inputs}
        filtered_results = filter_by_categorical_inputs(results, categorical_inputs, attribute_indices)
        results_to_hash = filtered_results
    else:
        results_to_hash = results

    # If review keywords are provided, perform LSH query
    if review_keywords and num_neighbors:
        review_index = list(data.columns).index('review')
        lsh_results = lsh_query(review_keywords.split(), num_neighbors, results_to_hash, review_index)
        # Extract the rows from the LSH results and keep all fields
        final_results = [(row + [cosine_sim]) for row, cosine_sim in lsh_results]
        return final_results
    else:
        return results_to_hash
