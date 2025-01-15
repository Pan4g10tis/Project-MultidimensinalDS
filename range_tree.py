import csv
from datetime import datetime
import time
from lsh import lsh_query


class Node(object):
    """A node in a Range Tree."""

    def __init__(self, value) -> None:
        self.value = value
        self.left = None
        self.right = None
        self.isLeaf = False
        self.assoc = None
        self.full_row = None


def category_reader(input_list, headings):
    dim = 0
    categories_table = []
    extras_table = []

    for item in input_list:
        while item not in headings:
            print("This category doesn't exist in the file: ", item)
            item = input("Please enter another category:")

        if item in ['100g_USD', 'rating', 'review_date']:
            dim += 1
            categories_table.append(item)
        else:
            extras_table.append(item)

    return categories_table, extras_table, dim


def date_to_numeric(date_str, reference_date="January 2017"):
    # Parse the input date and reference date
    date = datetime.strptime(date_str, "%B %Y")
    ref_date = datetime.strptime(reference_date, "%B %Y")

    # Calculate the number of months since the reference date
    numeric_value = (date.year - ref_date.year) * 12 + (date.month - ref_date.month)
    return numeric_value


def range_reader(categories, extras):
    range = []
    extra_range = []
    for i in categories:
        if i == '100g_USD':
            low = float(input("Give the first part of the range '100g_USD' (low): "))
            high = float(input("Give the second part of the range (high) '100g_USD' : "))
            if high == 0:
                high = 90.83
            range.append((low, high))

        elif i == 'rating':
            low = float(input("Give the first part of the range 'rating'(low) : "))
            high = float(input("Give the second part of the range 'rating'(high):  "))
            if high == 0:
                high = 97
            range.append((low, high))

        elif i == 'review_date':
            low = date_to_numeric(input("Give the first part of the range 'review_date' (low): "))
            high = date_to_numeric(input("Give the second part of the range 'review_date' (high): "))
            if high == 0:
                high = date_to_numeric('November 2022')
            range.append((low, high))

    for i in extras:
        print("Give the value you want for", i)
        value = input()
        extra_range.append(value)

    return range, extra_range


def load_data(filepath, categories):
    with open(filepath, encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        header = next(readCSV)

        # Identify the index of the categories
        indices = [header.index(category) for category in categories]
        data = []
        all_data = []
        for row in readCSV:
            if len(indices) == 1:
                if indices[0] == 7:  # Index of 'review_date'
                    # Convert 'review_date' to numeric format (YYYYMM)
                    date_str = row[indices[0]]
                    try:
                        date_numeric = int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
                    except ValueError:
                        date_numeric = 0  # Default value for invalid dates
                    data.append((date_numeric, row))
                else:
                    value = float(row[indices[0]]) if row[indices[0]] != '' else 0.0
                    data.append((value, row))
            else:
                # Extract relevant columns and store them as tuples
                values = []
                for i in indices:
                    if header[i].lower() == 'review_date':
                        # Convert 'review_date' to numeric format (YYYYMM)
                        date_str = row[i]
                        try:
                            date_numeric = int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
                        except ValueError:
                            date_numeric = 0  # Default value for invalid dates
                        values.append(date_numeric)
                    else:
                        value = float(row[i]) if row[i] != '' else 0.0
                        values.append(value)
                data.append((tuple(values), row))
            all_data.append(row)
    return data, all_data


def contains_comma(value):
    return ',' in value


def other_categories(search, headings, extra_range, extras):
    print("Extra Range:", extra_range)
    print("Extras:", extras)

    if len(extra_range) != len(extras):
        raise ValueError("Length of `extra_range` must match the length of `extras`!")

    filtered_results = []  # search  # Start with the full dataset

    for i, extra in enumerate(extras):  # Loop through the `extras` and `extra_range`
        if extra not in headings:  # Ensure the column exists
            raise ValueError(f"Column '{extra}' not found in headings!")

        # Find the index of the column in the headings
        col_index = headings.index(extra)

        if contains_comma(extra_range[i]):
            value = extra_range[i].split(',')
            value = [s.strip() for s in value]

            for j in value:
                temp_results = []
                temp_results = [
                    result for result in search
                    if result[col_index] == j
                ]
                filtered_results += temp_results
        else:
            # Filter based on the corresponding value in `extra_range`
            print(extra_range[i])
            filtered_results = [
                result for result in search
                if result[col_index] == extra_range[i]
            ]

    return filtered_results


def ConstructRangeTree1d(data):
    if not data:
        return None
    if len(data) == 1:
        value, row = data[0]
        node = Node(value)
        node.isLeaf = True
        node.full_row = row
    else:
        mid_val = len(data) // 2
        value, row = data[mid_val]
        node = Node(value)
        node.full_row = row
        node.left = ConstructRangeTree1d(data[:mid_val])
        node.right = ConstructRangeTree1d(data[mid_val + 1:])
    return node


def ConstructRangeTree2d(data, cur_dim=1):
    if not data:
        return None
    if len(data) == 1:
        value, row = data[0]
        node = Node(value)
        node.isLeaf = True
        node.full_row = row
    else:
        mid_val = len(data) // 2
        value, row = data[mid_val]
        node = Node(value)  # node.value = (x,y)
        node.full_row = row
        node.left = ConstructRangeTree2d(data[:mid_val], cur_dim)
        node.right = ConstructRangeTree2d(data[mid_val + 1:], cur_dim)
    if cur_dim == 1:
        node.assoc = ConstructRangeTree2d(sorted(data, key=lambda x: x[0][1]), cur_dim=2)
    return node


def ConstructRangeTree3d(data, cur_dim=1):
    if not data:
        return None
    if len(data) == 1:
        value, row = data[0]
        node = Node(value)
        node.isLeaf = True
        node.full_row = row
    else:
        mid_val = len(data) // 2
        value, row = data[mid_val]
        node = Node(value)  # node.value = (x,y,z)
        node.full_row = row
        node.left = ConstructRangeTree3d(data[:mid_val], cur_dim)
        node.right = ConstructRangeTree3d(data[mid_val + 1:], cur_dim)
    if cur_dim == 1:
        node.assoc = ConstructRangeTree3d(sorted(data, key=lambda x: x[0][1]), cur_dim=2)
    elif cur_dim == 2:
        node.assoc = ConstructRangeTree3d(sorted(data, key=lambda x: x[0][2]), cur_dim=3)
    return node


def withinRange(point, range, dim):
    if dim == 1:
        x = point
        return range[0][0] <= x <= range[0][1]
    elif dim == 2:
        x = point[0]
        y = point[1]
        return range[0][0] <= x <= range[0][1] and range[1][0] <= y <= range[1][1]
    elif dim == 3:
        x = point[0]
        y = point[1]
        z = point[2]
        return (range[0][0] <= x <= range[0][1] and
                range[1][0] <= y <= range[1][1] and
                range[2][0] <= z <= range[2][1])


def getValue(point, cur_dim, dim):
    value = point.value
    if dim == 1:
        return value
    elif dim == 2:
        if cur_dim == 1:
            return value[0]
        else:
            return value[1]
    elif dim == 3:
        if cur_dim == 1:
            return value[0]
        elif cur_dim == 2:
            return value[1]
        elif cur_dim == 3:
            return value[2]


def FindSplitNode(root, p_min, p_max, dim, cur_dim):
    splitnode = root
    while splitnode is not None:
        node = getValue(splitnode, cur_dim, dim)
        p_min = float(p_min)
        p_max = float(p_max)
        node = float(node)
        if p_max < node:
            splitnode = splitnode.left
        elif p_min > node:
            splitnode = splitnode.right
        elif p_min <= node <= p_max:
            break
    return splitnode


def SearchRangeTree1d(tree, p1, p2, dim, cur_dim=1):
    nodes = []
    splitnode = FindSplitNode(tree, p1, p2, dim, cur_dim)
    if splitnode is None:
        return nodes
    if withinRange(getValue(splitnode, cur_dim, dim), [(p1, p2)], 1):
        nodes.append(splitnode.full_row)
    nodes += SearchRangeTree1d(splitnode.left, p1, p2, dim, cur_dim)
    nodes += SearchRangeTree1d(splitnode.right, p1, p2, dim, cur_dim)
    return nodes


def SearchRangeTree2d(tree, x1, x2, y1, y2, dim, cur_dim=1):
    results = []
    splitnode = FindSplitNode(tree, x1, x2, dim, cur_dim)
    if splitnode is None:
        return results
    if withinRange(splitnode.value, [(x1, x2), (y1, y2)], dim):
        results.append(splitnode.full_row)
    vl = splitnode.left
    while vl is not None:
        if withinRange(vl.value, [(x1, x2), (y1, y2)], dim):
            results.append(vl.full_row)
        if x1 <= vl.value[0]:
            if vl.right is not None:
                results += SearchRangeTree1d(vl.right.assoc, y1, y2, dim, cur_dim + 1)
            vl = vl.left
        else:
            vl = vl.right
    vr = splitnode.right
    while vr is not None:
        if withinRange(vr.value, [(x1, x2), (y1, y2)], dim):
            results.append(vr.full_row)
        if x2 >= vr.value[0]:
            if vr.left is not None:
                results += SearchRangeTree1d(vr.left.assoc, y1, y2, dim, cur_dim + 1)
            vr = vr.right
        else:
            vr = vr.left
    return results


def SearchRangeTree3d(tree, x1, x2, y1, y2, z1, z2, dim, cur_dim=1):
    results = []
    splitnode = FindSplitNode(tree, x1, x2, dim, cur_dim)
    if splitnode is None:
        return results
    if withinRange(splitnode.value, [(x1, x2), (y1, y2), (z1, z2)], dim):
        results.append(splitnode.full_row)
    vl = splitnode.left
    while vl is not None:
        if withinRange(vl.value, [(x1, x2), (y1, y2), (z1, z2)], dim):
            results.append(vl.full_row)
        if x1 <= vl.value[0]:
            if vl.right is not None:
                results += SearchRangeTree2d(vl.right.assoc, y1, y2, z1, z2, dim, cur_dim + 1)
            vl = vl.left
        else:
            vl = vl.right
    vr = splitnode.right
    while vr is not None:
        if withinRange(vr.value, [(x1, x2), (y1, y2), (z1, z2)], dim):
            results.append(vr.full_row)
        if x2 >= vr.value[0]:
            if vr.left is not None:
                results += SearchRangeTree2d(vr.left.assoc, y1, y2, z1, z2, dim, cur_dim + 1)
            vr = vr.right
        else:
            vr = vr.left
    return results


def run_3d_batch_queries(file_path):
    """
    Process a batch of 3D queries from a file and execute them using a 3D range tree.

    Args:
        file_path (str): Path to the file containing batch queries.
    """
    global length
    headings = ['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD', 'rating', 'review_date', 'review']

    # Load 3D data: `100g_USD`, `rating`, and `review_date`
    categories = ['100g_USD', 'rating', 'review_date']
    data, all_data = load_data("simplified_coffee.csv", categories)
    data.sort()

    # Construct the 3D range tree
    tree = ConstructRangeTree3d(data)

    # Read the batch queries from the file
    with open(file_path, 'r') as f:
        queries = f.readlines()

    total_time = 0
    all_results = []

    for query in queries:
        # Parse the query line
        parts = query.strip().split(',')
        if len(parts) != 6:
            print(f"Skipping invalid query line: {query.strip()}")
            continue

        # Parse ranges for each dimension
        usd_min, usd_max = float(parts[0]), float(parts[1])
        rating_min, rating_max = float(parts[2]), float(parts[3])
        date_min, date_max = map(lambda date: date_to_numeric(date.strip()), parts[4:6])

        # Perform range query
        start = time.time()
        results = SearchRangeTree3d(tree,
                                    usd_min, usd_max,
                                    rating_min, rating_max,
                                    date_min, date_max,
                                    len(categories))
        end = time.time()

        # Store results and time
        query_time = end - start
        total_time += query_time
        all_results.append((query, results, query_time))

    # Output results
    print(f"Total time for all queries: {total_time:.4f} seconds")


def range_tree_main(selected_attributes=None, conditions=None, review_keywords=None, num_neighbors=None):
    """
    Main function for Range Tree search with LSH support for the 'review' column.
    :param selected_attributes: List of selected attributes (e.g., ["100g_USD", "rating"]).
    :param conditions: Dictionary of conditions for the selected attributes.
                      For numeric attributes, the value is a tuple (min_value, max_value).
                      For non-numeric attributes, the value is a string or list of strings.
    :param review_keywords: List of keywords for LSH-based search on the 'review' column.
    :param num_neighbors: Number of nearest neighbors to return for LSH-based search.
    :return: List of matching rows.
    """
    if selected_attributes is None:
        selected_attributes = []
    if conditions is None:
        conditions = {}

    print("Selected Attributes:", selected_attributes)
    print("Conditions:", conditions)
    print("Review Keywords:", review_keywords)
    print("Number of Nearest Neighbors:", num_neighbors)

    # File reading and formatting
    headings = ['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD', 'rating', 'review_date', 'review']
    print("Headings:", headings)

    # Separate numeric and non-numeric attributes
    numeric_attributes = [attr for attr in selected_attributes if attr in ['100g_USD', 'rating', 'review_date']]
    non_numeric_attributes = [attr for attr in selected_attributes if attr in ['roaster', 'roast', 'loc_country', 'origin']]
    print("Numeric Attributes:", numeric_attributes)
    print("Non-Numeric Attributes:", non_numeric_attributes)

    # Load data for numeric attributes only
    data, all_data = load_data("simplified_coffee.csv", numeric_attributes)
    print("Data loaded successfully.")

    # Extract numeric and categorical conditions
    numeric_ranges = {}
    categorical_inputs = {}

    for attr, value in conditions.items():
        if attr in numeric_attributes:
            # Numeric condition: value is a tuple (min_value, max_value)
            numeric_ranges[attr] = value
        elif attr in non_numeric_attributes:
            # Categorical condition: value is a string or list of strings
            if isinstance(value, str):
                categorical_inputs[attr] = [val.strip().lower() for val in value.split(",")]
            elif isinstance(value, list):
                categorical_inputs[attr] = [val.strip().lower() for val in value]

    print("Numeric Ranges:", numeric_ranges)
    print("Categorical Inputs:", categorical_inputs)

    # Initialize results
    results = []

    # Build and query the range tree only if numeric attributes are selected
    if numeric_attributes:
        # Build the range tree based on the number of numeric attributes
        dim = len(numeric_attributes)
        print("Dimension of Range Tree:", dim)
        data.sort()

        if dim == 1:
            tree = ConstructRangeTree1d(data)
        elif dim == 2:
            tree = ConstructRangeTree2d(data)
        elif dim == 3:
            tree = ConstructRangeTree3d(data)

        # Perform range query
        results = []
        if dim == 1:
            attr = numeric_attributes[0]
            min_val, max_val = numeric_ranges.get(attr, (None, None))
            if min_val is not None and max_val is not None:
                results = SearchRangeTree1d(tree, min_val, max_val, dim)
        elif dim == 2:
            attr1, attr2 = numeric_attributes
            min_val1, max_val1 = numeric_ranges.get(attr1, (None, None))
            min_val2, max_val2 = numeric_ranges.get(attr2, (None, None))
            if min_val1 is not None and max_val1 is not None and min_val2 is not None and max_val2 is not None:
                results = SearchRangeTree2d(tree, min_val1, max_val1, min_val2, max_val2, dim)
        elif dim == 3:
            attr1, attr2, attr3 = numeric_attributes
            min_val1, max_val1 = numeric_ranges.get(attr1, (None, None))
            min_val2, max_val2 = numeric_ranges.get(attr2, (None, None))
            min_val3, max_val3 = numeric_ranges.get(attr3, (None, None))
            if min_val1 is not None and max_val1 is not None and min_val2 is not None and max_val2 is not None and min_val3 is not None and max_val3 is not None:
                results = SearchRangeTree3d(tree, min_val1, max_val1, min_val2, max_val2, min_val3, max_val3, dim)

        print("Results from Range Query:", results)
    else:
        # If no numeric attributes are selected, use all data
        results = all_data
        print("No numeric attributes selected. Using all data.")

    # Filter results based on categorical conditions (if any)
    if categorical_inputs:
        filtered_results = []
        for result in results:
            match = True
            for attr, values in categorical_inputs.items():
                idx = headings.index(attr)
                if str(result[idx]).strip().lower() not in [v.lower() for v in values]:
                    match = False
                    break
            if match:
                filtered_results.append(result)
        results = filtered_results

    print("Filtered Results:", results)

    # Perform LSH-based search on the 'review' column if keywords are provided
    if review_keywords and num_neighbors:
        print("Performing LSH-based search on the 'review' column...")
        review_index = headings.index("review")  # Index of the 'review' column
        lsh_results = lsh_query(review_keywords, num_neighbors, results, review_index)
        results = lsh_results  # Replace results with LSH results
        print("LSH Results:", results)

    return results
