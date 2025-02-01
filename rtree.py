import pandas as pd
from datetime import datetime
from lsh import lsh_query


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
        node.entries.sort(key=lambda entry: entry[0].mins[0])
        mid = len(node.entries) // 2
        new_node = RTreeNode(is_leaf=node.is_leaf)
        new_node.entries = node.entries[mid:]
        node.entries = node.entries[:mid]

        def compute_bbox(entries):
            mins = [min(entry[0].mins[i] for entry in entries) for i in range(len(entries[0][0].mins))]
            maxs = [max(entry[0].maxs[i] for entry in entries) for i in range(len(entries[0][0].maxs))]
            return BoundingBox(mins, maxs)

        node_bbox = compute_bbox(node.entries)
        new_node_bbox = compute_bbox(new_node.entries)

        if node == self.root:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries.append((node_bbox, node))
            new_root.entries.append((new_node_bbox, new_node))
            self.root = new_root
        else:
            parent = self._find_parent(self.root, node)
            parent.entries.append((new_node_bbox, new_node))
            if parent.is_full(self.max_entries):
                self._split_node(parent)

    def _find_parent(self, current_node, target_node):
        """Find the parent of a given node."""
        if current_node.is_leaf:
            return None
        for entry in current_node.entries:
            bbox, child = entry
            if child == target_node:
                return current_node
            elif not child.is_leaf:
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
    """Check if the bounding box satisfies the numeric and non-numeric conditions."""
    for idx, attr in enumerate(selected_numeric):
        if attr in parsed_conditions:
            min_value = bbox.mins[idx]
            max_value = bbox.maxs[idx]
            for op, val in parsed_conditions[attr]:
                if op == ">=" and max_value < val:
                    return False
                elif op == "<=" and min_value > val:
                    return False
                elif op == ">" and max_value <= val:
                    return False
                elif op == "<" and min_value >= val:
                    return False

    if row is not None:
        for attr, condition_list in non_numeric_conditions.items():
            if attr in row:
                attr_value = str(row[attr]).strip().lower()
                if isinstance(condition_list, list) and attr_value not in condition_list:
                    return False
            else:
                return False
    return True


def search_node(node, data, selected_numeric, parsed_conditions, non_numeric_conditions, matching_entries):
    """Recursive function to search the R-tree."""
    if node.is_leaf:
        for bbox, obj in node.entries:
            row = data.loc[obj]
            if satisfies_conditions(bbox, row, selected_numeric, parsed_conditions, non_numeric_conditions):
                matching_entries.append(obj)
    else:
        for bbox, child in node.entries:
            if satisfies_conditions(bbox, None, selected_numeric, parsed_conditions, non_numeric_conditions):
                search_node(child, data, selected_numeric, parsed_conditions, non_numeric_conditions, matching_entries)


def rtree_main(selected_attributes, conditions, review_keywords=None, num_neighbors=None):
    """Main function for R-tree search with LSH integration."""
    data = pd.read_csv("simplified_coffee.csv")
    data["review_date"] = data["review_date"].apply(convert_date_to_numeric)

    numeric_attributes = ["100g_USD", "rating", "review_date"]
    selected_numeric = [attr for attr in selected_attributes if attr in numeric_attributes]
    selected_non_numeric = [attr for attr in selected_attributes if attr not in numeric_attributes]

    r_tree = None
    if selected_numeric:
        r_tree = RTree(max_entries=5)
        for idx, row in data.iterrows():
            mins = [row[attr] for attr in selected_numeric]
            maxs = mins[:]
            bbox = BoundingBox(mins=mins, maxs=maxs)
            r_tree.insert(bbox, idx)

    parsed_conditions = {}
    non_numeric_conditions = {}

    for attr, condition_list in conditions.items():
        if attr in selected_numeric:
            parsed_conditions[attr] = []
            for condition in condition_list:
                if condition[:2] in [">=", "<="]:
                    operator = condition[:2]
                    value = condition[2:].strip()
                elif condition[0] in [">", "<"]:
                    operator = condition[0]
                    value = condition[1:].strip()
                else:
                    continue
                value = float(value)
                parsed_conditions[attr].append((operator, value))
        elif attr in selected_non_numeric:
            non_numeric_conditions[attr] = []
            for condition in condition_list:
                condition = condition.strip().replace("'", "").replace('"', "").strip()
                if " OR " in condition:
                    non_numeric_conditions[attr].extend([val.strip().lower() for val in condition.split(" OR ")])
                else:
                    non_numeric_conditions[attr].append(condition.strip().lower())

    matching_entries = []
    if selected_numeric:
        search_node(r_tree.root, data, selected_numeric, parsed_conditions, non_numeric_conditions, matching_entries)
    else:
        for idx, row in data.iterrows():
            if all(satisfies_conditions(None, row, selected_numeric, parsed_conditions, non_numeric_conditions)
                   for attr, condition in non_numeric_conditions.items()):
                matching_entries.append(idx)

    if matching_entries:
        matching_rows = data.loc[matching_entries, ['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD',
                                                    'rating', 'review_date', 'review']]
        if review_keywords:
            review_index = matching_rows.columns.get_loc("review")
            lsh_results = lsh_query(review_keywords.split(), num_neighbors, matching_rows.values.tolist(), review_index)
            matching_rows = [row for row, _ in lsh_results]
            matching_rows = pd.DataFrame(matching_rows, columns=['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD', 'rating', 'review_date', 'review'])
        return [tuple(row) for row in matching_rows.itertuples(index=False)]
    return []
