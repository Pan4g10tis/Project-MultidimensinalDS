import pandas as pd
import math
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


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
    filtered_results = []
    for result in results:
        match = True
        for attribute, values in categorical_inputs.items():
            idx = attribute_indices[attribute]
            if result[idx] not in values:
                match = False
                break
        if match:
            filtered_results.append(result)

    return filtered_results


# Main Function
def octree_main():
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

    # Perform Query
    results = octree.range_query(range_min, range_max)

    if categorical_inputs:
        attribute_indices = {attribute: list(data.columns).index(attribute) for attribute in categorical_inputs}
        filtered_results = filter_by_categorical_inputs(results, categorical_inputs, attribute_indices)
        results_to_hash = filtered_results
        print(f"Found {len(filtered_results)} results within the specified range and categorical inputs:")
        for result in filtered_results:
            print(result)
    else:
        results_to_hash = results
        print(f"Found {len(results)} results within the specified range:")
        for result in results:
            print(result)


# Uncomment to run
octree_main()
