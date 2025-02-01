import csv
from datetime import datetime
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


def date_to_numeric(date_str, reference_date="January 2017"):
    date = datetime.strptime(date_str, "%B %Y")
    ref_date = datetime.strptime(reference_date, "%B %Y")
    numeric_value = (date.year - ref_date.year) * 12 + (date.month - ref_date.month)
    return numeric_value


def load_data(filepath, categories):
    with open(filepath, encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        header = next(readCSV)

        indices = [header.index(category) for category in categories]
        data = []
        all_data = []
        for row in readCSV:
            if len(indices) == 1:
                if indices[0] == 7:  # Index of 'review_date'
                    date_str = row[indices[0]]
                    try:
                        date_numeric = int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
                    except ValueError:
                        date_numeric = 0
                    data.append((date_numeric, row))
                else:
                    value = float(row[indices[0]]) if row[indices[0]] != '' else 0.0
                    data.append((value, row))
            else:
                values = []
                for i in indices:
                    if header[i].lower() == 'review_date':
                        date_str = row[i]
                        try:
                            date_numeric = int(datetime.strptime(date_str, "%B %Y").strftime("%Y%m"))
                        except ValueError:
                            date_numeric = 0
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
    if len(extra_range) != len(extras):
        raise ValueError("Length of `extra_range` must match the length of `extras`!")

    filtered_results = []
    for i, extra in enumerate(extras):
        if extra not in headings:
            raise ValueError(f"Column '{extra}' not found in headings!")

        col_index = headings.index(extra)
        if contains_comma(extra_range[i]):
            value = extra_range[i].split(',')
            value = [s.strip() for s in value]
            for j in value:
                temp_results = [result for result in search if result[col_index] == j]
                filtered_results += temp_results
        else:
            filtered_results = [result for result in search if result[col_index] == extra_range[i]]
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
        node = Node(value)
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
        node = Node(value)
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
        x, y = point
        return range[0][0] <= x <= range[0][1] and range[1][0] <= y <= range[1][1]
    elif dim == 3:
        x, y, z = point
        return range[0][0] <= x <= range[0][1] and range[1][0] <= y <= range[1][1] and range[2][0] <= z <= range[2][1]


def getValue(point, cur_dim, dim):
    value = point.value
    if dim == 1:
        return value
    elif dim == 2:
        return value[0] if cur_dim == 1 else value[1]
    elif dim == 3:
        return value[0] if cur_dim == 1 else value[1] if cur_dim == 2 else value[2]


def FindSplitNode(root, p_min, p_max, dim, cur_dim):
    splitnode = root
    while splitnode is not None:
        node = getValue(splitnode, cur_dim, dim)
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


def range_tree_main(selected_attributes=None, conditions=None, review_keywords=None, num_neighbors=None):
    if selected_attributes is None:
        selected_attributes = []
    if conditions is None:
        conditions = {}

    headings = ['name', 'roaster', 'roast', 'loc_country', 'origin', '100g_USD', 'rating', 'review_date', 'review']
    numeric_attributes = [attr for attr in selected_attributes if attr in ['100g_USD', 'rating', 'review_date']]
    non_numeric_attributes = [attr for attr in selected_attributes if attr in ['roaster', 'roast', 'loc_country', 'origin']]

    data, all_data = load_data("simplified_coffee.csv", numeric_attributes)

    numeric_ranges = {}
    categorical_inputs = {}
    for attr, value in conditions.items():
        if attr in numeric_attributes:
            numeric_ranges[attr] = value
        elif attr in non_numeric_attributes:
            if isinstance(value, str):
                categorical_inputs[attr] = [val.strip().lower() for val in value.split(",")]
            elif isinstance(value, list):
                categorical_inputs[attr] = [val.strip().lower() for val in value]

    results = []
    if numeric_attributes:
        dim = len(numeric_attributes)
        data.sort()
        if dim == 1:
            tree = ConstructRangeTree1d(data)
        elif dim == 2:
            tree = ConstructRangeTree2d(data)
        elif dim == 3:
            tree = ConstructRangeTree3d(data)

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
    else:
        results = all_data

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

    if review_keywords and num_neighbors:
        review_index = headings.index("review")
        lsh_input = [list(row) for row in results]
        lsh_results = lsh_query({review_keywords}, num_neighbors, lsh_input, review_index)
        results = [row for row, _ in lsh_results]

    return results
