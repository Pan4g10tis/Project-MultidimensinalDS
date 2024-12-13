import pandas as pd
from datetime import datetime
import time


def rtree_main():

    data = pd.read_csv("simplified_coffee.csv")

    def convert_date_to_numeric(date_str):
        """Convert 'Month Year' date format to numeric YYYYMM format."""
        try:
            return int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")

    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    available_columns = [col for col in data.columns if col != "name" and col != "review"]

    # Display available columns for selection
    print("Available attributes for indexing:")
    for i, col in enumerate(available_columns):
        print(f"{i + 1}. {col}")

    # Allow the user to choose up to 4 attributes
    while True:
        try:
            # User input: indices of attributes to select
            selected_indices = input("Enter the numbers of up to 4 attributes you want to index, "
                                     "separated by commas (e.g., 2,3,5,6): ")
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

    # Identify numeric and non-numeric attributes
    numeric_attributes = ["100g_USD", "rating", "review_date"]
    selected_numeric = [attr for attr in selected_attributes if attr in numeric_attributes]
    selected_non_numeric = [attr for attr in selected_attributes if attr not in numeric_attributes]

    # Display classification
    print(f"Numeric attributes for R-tree: {selected_numeric}")
    print(f"Non-numeric attributes (excluded from R-tree): {selected_non_numeric}")

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

    # Redefine the parsing of conditions to store tuples of the operator and value.
    def parse_condition(user_input):
        """Parse the user condition input and store the operator and value."""
        try:
            operator, value = user_input[:2], user_input[2:].strip()
            if operator in [">=", "<=", ">", "<"]:
                return operator, float(value)
            else:
                raise ValueError(f"Invalid operator: {operator}")
        except Exception as e:
            print(f"Invalid condition format: {e}")
            return None

    # Collect user-defined conditions for numeric attributes if selected
    conditions = {}
    if selected_numeric:
        print("Specify search conditions for the numeric attributes you selected for indexing:")

        for attr in selected_numeric:
            user_conditions = []
            print(
                f"Enter conditions for {attr} (e.g., >93, <=10, etc. for rating and price (USD per 100g), "
                f"or >201711 for review date (YYYYMM))"
            )
            while True:
                user_condition = input(f"Enter condition for {attr} (or press Enter to finish): ").strip()
                if not user_condition:
                    break  # Finish entering conditions
                condition = parse_condition(user_condition)
                if condition:
                    user_conditions.append(condition)
                else:
                    print(f"Invalid condition format for {attr}: {user_condition}. Skipping this condition.")

            if user_conditions:
                conditions[attr] = user_conditions

    # Collect user-defined conditions for non-numeric attributes if selected
    non_numeric_conditions = {}
    if selected_non_numeric:
        print("Specify search conditions for the non-numeric attributes you selected for indexing:")
        for attr in selected_non_numeric:
            user_condition = input(
                f"Enter condition for {attr} (e.g., exact match 'value', contains 'substring', "
                f"or list of values separated by OR, like 'value1 OR value2'): ").strip()
            if user_condition:
                # Remove quotes if present
                user_condition = user_condition.replace("'", "").replace('"', "").strip()
                if " OR " in user_condition:
                    # Split on " OR " and store as a list
                    non_numeric_conditions[attr] = [val.strip().lower() for val in user_condition.split(" OR ")]
                else:
                    # Store single condition as a string
                    non_numeric_conditions[attr] = user_condition.strip().lower()

    # Validate that at least one condition is valid
    if not conditions and not non_numeric_conditions:
        print("No valid conditions provided. Cannot perform search.")
        exit()

    # Define a function to test if a bounding box satisfies the conditions
    def satisfies_conditions(bbox, row=None):
        """
        Check if the bounding box satisfies the numeric conditions,
        and the row satisfies the non-numeric conditions (if provided).
        """
        # Check numeric conditions
        for idx, attr in enumerate(selected_numeric):
            if attr in conditions:
                min_value = bbox.mins[idx]
                max_value = bbox.maxs[idx]
                valid = True
                for op, val in conditions[attr]:
                    if op == ">=":
                        if max_value < val:
                            valid = False
                            break
                    elif op == "<=":
                        if min_value > val:
                            valid = False
                            break
                    elif op == ">":
                        if max_value <= val:
                            valid = False
                            break
                    elif op == "<":
                        if min_value >= val:
                            valid = False
                            break
                if not valid:
                    return False

        # Check non-numeric conditions (for rows only, i.e., leaf nodes)
        if row is not None:
            for attr, condition in non_numeric_conditions.items():
                if attr in row:
                    attr_value = str(row[attr]).strip().lower()
                    if isinstance(condition, list):  # Multi-value condition
                        # Check if attr_value matches any value in the list
                        if attr_value not in condition:
                            return False
                    elif condition.startswith("exact match"):
                        match_value = condition.replace("exact match '", "").replace("'", "").strip().lower()
                        if attr_value != match_value:
                            return False
                    elif condition.startswith("contains"):
                        substring = condition.replace("contains '", "").replace("'", "").strip().lower()
                        if substring not in attr_value:
                            return False
                    else:
                        print(f"Unrecognized condition format for {attr}: {condition}")
                        return False
                else:
                    return False  # Attribute not found in the row

        return True

    def search_node(node):
        """Recursive function to search the R-tree, considering both numeric and non-numeric conditions."""
        if node.is_leaf:
            for bbox, obj in node.entries:
                row = data.loc[obj]  # Get the corresponding row
                if satisfies_conditions(bbox, row):  # Pass the row for leaf nodes
                    matching_entries.append(obj)
        else:
            for bbox, child in node.entries:
                if satisfies_conditions(bbox):  # For internal nodes, we only check the bounding box
                    search_node(child)

    matching_entries = []

    # Normalize non-numeric data for comparison
    for attr in selected_non_numeric:
        if attr in data.columns:
            data[attr] = data[attr].astype(str).str.strip().str.lower()

    if selected_numeric:
        start = time.time()
        # Start the search from the root node
        search_node(r_tree.root)
        end = time.time()
        length = end - start
    else:
        # Perform linear search based on non-numeric attributes
        for idx, row in data.iterrows():
            if all(
                    satisfies_conditions(None, row)  # Use satisfies_conditions for all non-numeric checks
                    for attr, condition in non_numeric_conditions.items()
            ):
                matching_entries.append(idx)

    # Fetch matching rows from the dataset
    if matching_entries:
        matching_rows = data.loc[matching_entries, ['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD',
                                                    'rating', 'review_date']]
        print("Search Results:")
        print(matching_rows)
    else:
        print("No results match the given conditions.")

    print(length)

    def run_batch_queries(file_path):
        with open(file_path, 'r') as file:
            queries = file.readlines()

        total_time = 0
        matching_entries = []

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
            search_node(r_tree.root)
            end = time.time()

            total_time += (end - start)
            matching_entries.extend(matching_entries)

        print(f"Total search time for batch queries: {total_time} seconds")
        print(f"Total matching entries: {len(set(matching_entries))}")

    run_batch_queries("queries.txt")


rtree_main()
