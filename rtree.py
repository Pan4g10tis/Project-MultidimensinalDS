import pandas as pd
from datetime import datetime
import time


# Define R-tree classes
class BoundingBox:
    """Represents a bounding box in the R-tree."""
    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def overlaps(self, other):
        """Check if this bounding box overlaps with another."""
        return all(self.mins[i] <= other.maxs[i] and self.maxs[i] >= other.mins[i] for i in range(len(self.mins)))


class RTreeNode:
    """Node in the R-tree."""
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []  # Holds bounding boxes and children/objects

    def is_full(self, max_entries):
        """Check if the node is full."""
        return len(self.entries) >= max_entries


class RTree:
    """R-tree implementation."""
    def __init__(self, max_entries=5):
        self.max_entries = max_entries
        self.root = RTreeNode()

    def insert(self, bbox, obj):
        """Insert a bounding box and associated object into the R-tree."""
        node = self._choose_leaf(self.root, bbox)
        node.entries.append((bbox, obj))
        if node.is_full(self.max_entries):
            self._split_node(node)

    def _choose_leaf(self, node, bbox):
        """Choose the appropriate leaf node for insertion."""
        if node.is_leaf:
            return node
        # For non-leaf nodes, choose the entry with the minimum enlargement
        best_entry = min(node.entries, key=lambda entry: self._calculate_enlargement(entry[0], bbox))
        return self._choose_leaf(best_entry[1], bbox)

    def _calculate_enlargement(self, node_bbox, new_bbox):
        """Calculate the enlargement needed to include new_bbox in node_bbox."""
        new_mins = [min(node_bbox.mins[i], new_bbox.mins[i]) for i in range(len(node_bbox.mins))]
        new_maxs = [max(node_bbox.maxs[i], new_bbox.maxs[i]) for i in range(len(node_bbox.maxs))]
        old_area = self._calculate_area(node_bbox)
        new_area = self._calculate_area(BoundingBox(new_mins, new_maxs))
        return new_area - old_area

    def _calculate_area(self, bbox):
        """Calculate the area of a bounding box."""
        return sum(bbox.maxs[i] - bbox.mins[i] for i in range(len(bbox.mins)))

    def _split_node(self, node):
        """Split a node that exceeds max_entries."""
        # Sort entries by the first dimension for simplicity
        node.entries.sort(key=lambda entry: entry[0].mins[0])

        # Divide entries into two groups
        mid = len(node.entries) // 2
        new_node = RTreeNode(is_leaf=node.is_leaf)
        new_node.entries = node.entries[mid:]
        node.entries = node.entries[:mid]

        # Create a bounding box for each node
        def compute_bbox(entries):
            mins = [min(entry[0].mins[i] for entry in entries) for i in range(len(entries[0][0].mins))]
            maxs = [max(entry[0].maxs[i] for entry in entries) for i in range(len(entries[0][0].maxs))]
            return BoundingBox(mins, maxs)

        # Bounding boxes for the split nodes
        node_bbox = compute_bbox(node.entries)
        new_node_bbox = compute_bbox(new_node.entries)

        # If splitting the root
        if node == self.root:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries.append((node_bbox, node))
            new_root.entries.append((new_node_bbox, new_node))
            self.root = new_root
        else:
            # Add the new node to the parent
            parent = self._find_parent(self.root, node)
            parent.entries.append((new_node_bbox, new_node))

            # If parent is full, recursively split it
            if parent.is_full(self.max_entries):
                self._split_node(parent)

    def _find_parent(self, current_node, target_node):
        """Find the parent of a given node."""
        if current_node.is_leaf:
            return None  # Leaf nodes have no children, so no parent

        for entry in current_node.entries:
            bbox, child = entry
            if child == target_node:
                return current_node
            elif not child.is_leaf:
                # Recursively search in non-leaf children
                parent = self._find_parent(child, target_node)
                if parent:
                    return parent
        return None


def convert_date_to_numeric(date_str):
    """Convert 'Month Year' date format to numeric YYYYMM format."""
    try:
        return int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")


def satisfies_conditions(bbox, row, selected_numeric, parsed_conditions, non_numeric_conditions):
    """
    Check if the bounding box satisfies the numeric conditions,
    and the row satisfies the non-numeric conditions (if provided).
    """
    # Check numeric conditions
    for idx, attr in enumerate(selected_numeric):
        if attr in parsed_conditions:
            min_value = bbox.mins[idx]
            max_value = bbox.maxs[idx]
            for op, val in parsed_conditions[attr]:  # parsed_conditions[attr] is a list of tuples
                if op == ">=":
                    if max_value < val:
                        return False
                elif op == "<=":
                    if min_value > val:
                        return False
                elif op == ">":
                    if max_value <= val:
                        return False
                elif op == "<":
                    if min_value >= val:
                        return False

    # Check non-numeric conditions (for rows only, i.e., leaf nodes)
    if row is not None:
        for attr, condition_list in non_numeric_conditions.items():
            if attr in row:
                attr_value = str(row[attr]).strip().lower()
                if isinstance(condition_list, list):  # Multi-value condition
                    # Check if attr_value matches any value in the list
                    if attr_value not in condition_list:
                        return False
                else:
                    print(f"Unrecognized condition format for {attr}: {condition_list}")
                    return False
            else:
                return False  # Attribute not found in the row

    return True


def search_node(node, data, selected_numeric, parsed_conditions, non_numeric_conditions, matching_entries):
    """Recursive function to search the R-tree, considering both numeric and non-numeric conditions."""
    if node.is_leaf:
        for bbox, obj in node.entries:
            row = data.loc[obj]  # Get the corresponding row
            if satisfies_conditions(bbox, row, selected_numeric, parsed_conditions, non_numeric_conditions):
                matching_entries.append(obj)
    else:
        for bbox, child in node.entries:
            if satisfies_conditions(bbox, None, selected_numeric, parsed_conditions, non_numeric_conditions):
                search_node(child, data, selected_numeric, parsed_conditions, non_numeric_conditions, matching_entries)


def rtree_main(selected_attributes, conditions):
    """
    Main function for R-tree search.
    :param selected_attributes: List of selected attributes (e.g., ["100g_USD", "rating"]).
    :param conditions: Dictionary of conditions for the selected attributes (e.g., {"rating": [">93.0", "<96.0"]}).
    :return: Tuple containing the R-tree object and the matching rows.
    """
    data = pd.read_csv("simplified_coffee.csv")

    # Convert review_date to numeric format
    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    # Identify numeric and non-numeric attributes
    numeric_attributes = ["100g_USD", "rating", "review_date"]
    selected_numeric = [attr for attr in selected_attributes if attr in numeric_attributes]
    selected_non_numeric = [attr for attr in selected_attributes if attr not in numeric_attributes]

    # Display classification
    print(f"Numeric attributes for R-tree: {selected_numeric}")
    print(f"Non-numeric attributes (excluded from R-tree): {selected_non_numeric}")

    # Initialize r_tree to None
    r_tree = None

    if not selected_numeric:
        print("No numeric attributes selected for R-tree. Search will be performed linearly"
              " based on non-numeric attributes.")
    else:
        # Create the R-tree
        r_tree = RTree(max_entries=5)

        # Insert data into the R-tree
        for idx, row in data.iterrows():
            mins = [row[attr] for attr in selected_numeric]
            maxs = mins[:]  # Single point, so mins and maxs are identical
            bbox = BoundingBox(mins=mins, maxs=maxs)
            r_tree.insert(bbox, idx)

        print("R-tree has been successfully built with the selected numeric attributes.")

    run_batch_queries("queries.txt", r_tree, data)

    # Parse conditions
    parsed_conditions = {}
    non_numeric_conditions = {}

    for attr, condition_list in conditions.items():
        if attr in selected_numeric:
            parsed_conditions[attr] = []
            for condition in condition_list:
                try:
                    # Check for two-character operators (>=, <=)
                    if condition[:2] in [">=", "<="]:
                        operator = condition[:2]
                        value = condition[2:].strip()
                    # Check for single-character operators (>, <)
                    elif condition[0] in [">", "<"]:
                        operator = condition[0]
                        value = condition[1:].strip()
                    else:
                        print(f"Invalid operator in condition for {attr}: {condition}")
                        continue

                    # Convert the value to a float
                    value = float(value)
                    parsed_conditions[attr].append((operator, value))
                except Exception as e:
                    print(f"Invalid condition format for {attr}: {condition}. Error: {e}")
                    continue
        elif attr in selected_non_numeric:
            # Handle non-numeric conditions
            non_numeric_conditions[attr] = []
            for condition in condition_list:
                condition = condition.strip()
                if condition:
                    # Remove quotes if present
                    condition = condition.replace("'", "").replace('"', "").strip()
                    if " OR " in condition:
                        # Split on " OR " and store as a list
                        non_numeric_conditions[attr].extend([val.strip().lower() for val in condition.split(" OR ")])
                    else:
                        # Store single condition as a string
                        non_numeric_conditions[attr].append(condition.strip().lower())

    # Validate that at least one condition is valid
    if not parsed_conditions and not non_numeric_conditions:
        print("No valid conditions provided. Cannot perform search.")
        return r_tree, []

    matching_entries = []

    # Normalize non-numeric data for comparison
    for attr in selected_non_numeric:
        if attr in data.columns:
            data[attr] = data[attr].astype(str).str.strip().str.lower()

    # Initialize length to None
    length = None

    if selected_numeric:
        start = time.time()
        # Start the search from the root node
        search_node(r_tree.root, data, selected_numeric, parsed_conditions, non_numeric_conditions, matching_entries)
        end = time.time()
        length = end - start
    else:
        # Perform linear search based on non-numeric attributes
        start = time.time()
        for idx, row in data.iterrows():
            if all(
                    satisfies_conditions(None, row, selected_numeric, parsed_conditions, non_numeric_conditions)
                    for attr, condition in non_numeric_conditions.items()
            ):
                matching_entries.append(idx)
        end = time.time()
        length = end - start

    # Fetch matching rows from the dataset
    if matching_entries:
        matching_rows = data.loc[matching_entries, ['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD',
                                                    'rating', 'review_date']]
        print("Search Results:")
        print(matching_rows)
        # Convert the DataFrame to a list of tuples
        return [tuple(row) for row in matching_rows.itertuples(index=False)]
    else:
        print("No results match the given conditions.")
        return []

    # Print the search time
    if length is not None:
        print(f"Search time: {length:.4f} seconds")
    else:
        print("Search time: N/A")


def run_batch_queries(file_path, r_tree, data):
    """
    Run batch queries from a file.
    :param file_path: Path to the file containing batch queries.
    :param r_tree: The R-tree object to use for searching.
    :param data: The dataset to search through.
    """
    with open(file_path, 'r') as file:
        queries = file.readlines()

    total_time = 0
    all_matching_entries = set()  # Use a set to avoid duplicates
    non_numeric_conditions = {}

    for query in queries:
        attributes = query.strip().split(',')

        search_bounds = [
            [float(attributes[0]), float(attributes[1])],  # Bounds for 100g_USD
            [float(attributes[2]), float(attributes[3])]  # Bounds for rating
        ]
        date_range = [
            convert_date_to_numeric(attributes[4]),  # Start date
            convert_date_to_numeric(attributes[5])  # End date
        ]

        bbox = BoundingBox(
            mins=[search_bounds[0][0], search_bounds[1][0], date_range[0]],
            maxs=[search_bounds[0][1], search_bounds[1][1], date_range[1]]
        )

        start = time.time()
        if r_tree is not None:
            matching_entries = []
            search_node(r_tree.root, data, ["100g_USD", "rating", "review_date"], {}, {}, matching_entries)
            all_matching_entries.update(matching_entries)
        else:
            # Perform linear search if no R-tree is available
            for idx, row in data.iterrows():
                if all(
                        satisfies_conditions(None, row, ["100g_USD", "rating", "review_date"], {}, {})
                        for attr, condition in non_numeric_conditions.items()
                ):
                    all_matching_entries.add(idx)
        end = time.time()

        total_time += (end - start)

    print(f"Total search time for batch queries: {total_time} seconds")
    print(f"Total matching entries: {len(all_matching_entries)}")
