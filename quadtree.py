import pandas as pd
import math
import time
from datetime import datetime


# Helper to convert date to numeric
def convert_date_to_numeric(date_str):
    try:
        return int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")


# Octree Node Class
class OctreeNode:
    def __init__(self, bounds, capacity=4):
        self.bounds = bounds  # Bounds: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        self.capacity = capacity
        self.points = []  # Stores points within the bounds
        self.children = None  # Eight child nodes after splitting

    def is_within_bounds(self, point):
        """Check if a point lies within the node's bounds."""
        for i in range(3):
            if not (self.bounds[i][0] <= point[i] <= self.bounds[i][1]):
                return False
        return True

    def split(self):
        """Split the node into 8 children with equal sub-bounds."""
        x_mid = (self.bounds[0][0] + self.bounds[0][1]) / 2
        y_mid = (self.bounds[1][0] + self.bounds[1][1]) / 2
        z_mid = (self.bounds[2][0] + self.bounds[2][1]) / 2

        sub_bounds = [
            [[self.bounds[0][0], x_mid], [self.bounds[1][0], y_mid], [self.bounds[2][0], z_mid]],  # Bottom-left-front
            [[x_mid, self.bounds[0][1]], [self.bounds[1][0], y_mid], [self.bounds[2][0], z_mid]],  # Bottom-right-front
            [[self.bounds[0][0], x_mid], [y_mid, self.bounds[1][1]], [self.bounds[2][0], z_mid]],  # Top-left-front
            [[x_mid, self.bounds[0][1]], [y_mid, self.bounds[1][1]], [self.bounds[2][0], z_mid]],  # Top-right-front
            [[self.bounds[0][0], x_mid], [self.bounds[1][0], y_mid], [z_mid, self.bounds[2][1]]],  # Bottom-left-back
            [[x_mid, self.bounds[0][1]], [self.bounds[1][0], y_mid], [z_mid, self.bounds[2][1]]],  # Bottom-right-back
            [[self.bounds[0][0], x_mid], [y_mid, self.bounds[1][1]], [z_mid, self.bounds[2][1]]],  # Top-left-back
            [[x_mid, self.bounds[0][1]], [y_mid, self.bounds[1][1]], [z_mid, self.bounds[2][1]]],  # Top-right-back
        ]

        self.children = [OctreeNode(bounds, self.capacity) for bounds in sub_bounds]

    def insert(self, point, full_data):
        """Insert a point into the Octree."""
        if not self.is_within_bounds(point):
            return False

        if self.children is None:
            if len(self.points) < self.capacity:
                self.points.append((point, full_data))
                return True
            else:
                self.split()

        # Insert into child nodes
        for child in self.children:
            if child.insert(point, full_data):
                return True

        return False

    def range_query(self, range_min, range_max, results=None):
        """Perform a range query and collect results."""
        if results is None:
            results = []

        # Check if the current node intersects with the range
        for i in range(3):
            if self.bounds[i][1] < range_min[i] or self.bounds[i][0] > range_max[i]:
                return results

        # Check points within the current node
        for point, full_data in self.points:
            if all(range_min[i] <= point[i] <= range_max[i] for i in range(3)):
                results.append(full_data)

        # Query child nodes if they exist
        if self.children is not None:
            for child in self.children:
                child.range_query(range_min, range_max, results)

        return results


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
            dataset_value = str(result[idx]).strip().lower()  # Normalize dataset value
            search_values = [val.strip().lower() for val in values]  # Normalize search values
            if dataset_value not in search_values:
                match = False
                break
        if match:
            filtered_results.append(result)
    return filtered_results


def run_batch_queries(query_file, octree, data):
    total_time = 0

    with open(query_file, "r") as file:
        queries = file.readlines()

    for query in queries:
        try:
            parts = query.strip().split(',')
            if len(parts) != 6:
                print(f"Skipping invalid query: {query.strip()}")
                continue

            usd_min, usd_max = float(parts[0]), float(parts[1])
            rating_min, rating_max = float(parts[2]), float(parts[3])
            date_min, date_max = map(lambda date: convert_date_to_numeric(date.strip()), parts[4:6])

            range_min = [usd_min, rating_min, date_min]
            range_max = [usd_max, rating_max, date_max]

            range_min = [x if x is not None else -math.inf for x in range_min]
            range_max = [x if x is not None else math.inf for x in range_max]

            start = time.time()
            results = octree.range_query(range_min, range_max)
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

    # Build Octree
    columns_for_splitting = ['100g_USD', 'rating', 'review_date']
    points = list(data[columns_for_splitting].to_records(index=False))
    full_data = data.values.tolist()

    bounds = [
        [data['100g_USD'].min(), data['100g_USD'].max()],
        [data['rating'].min(), data['rating'].max()],
        [data['review_date'].min(), data['review_date'].max()],
    ]

    octree = OctreeNode(bounds)

    for point, row in zip(points, full_data):
        octree.insert(point, row)

    # Run batch queries
    total_execution_time = run_batch_queries(query_file, octree, data)
    print(f"Total execution time for all queries: {total_execution_time:.4f} seconds")


# Main Function
def octree_main(selected_attributes=None, conditions=None):
    if selected_attributes is None:
        selected_attributes = []
    if conditions is None:
        conditions = {}

    data = pd.read_csv("simplified_coffee.csv")
    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    # Prepare data for Octree
    columns_for_splitting = ['100g_USD', 'rating', 'review_date']
    points = list(data[columns_for_splitting].to_records(index=False))
    full_data = data.values.tolist()

    # Define overall bounds
    bounds = [
        [data['100g_USD'].min(), data['100g_USD'].max()],
        [data['rating'].min(), data['rating'].max()],
        [data['review_date'].min(), data['review_date'].max()],
    ]

    # Build Octree
    octree = OctreeNode(bounds)

    for point, row in zip(points, full_data):
        octree.insert(point, row)

    # Define the type of each attribute
    numeric_attributes = ['100g_USD', 'rating', 'review_date']
    numeric_ranges = {}

    categorical_attributes = ['roaster', 'roast', 'loc_country', 'origin']
    categorical_inputs = {}

    # Extract numeric and categorical conditions
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
    results = octree.range_query(range_min, range_max)
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
