def get_row(lst, x):
    """
    You are given a 2 dimensional data, as a nested lists,
    which is similar to matrix, however, unlike matrices,
    each row may contain a different number of columns.
    Given lst, and integer x, find integers x in the list,
    and return list of tuples, [(x1, y1), (x2, y2) ...] such that
    each tuple is a coordinate - (row, columns), starting with 0.
    Sort coordinates initially by rows in ascending order.
    Also, sort coordinates of the row by columns in descending order.
    
    Examples:
    get_row([
      [1,2,3,4,5,6],
      [1,2,3,4,1,6],
      [1,2,3,4,5,1]
    ], 1) == [(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]
    get_row([], 1) == []
    get_row([[], [1], [1, 2, 3]], 3) == [(2, 2)]
    """
    coordinates = []
    
    # Iterate through each row with its index
    for row_index, row in enumerate(lst):
        # Iterate through each element in the row with its index
        for col_index, element in enumerate(row):
            # If we find the target value x, record its coordinates
            if element == x:
                coordinates.append((row_index, col_index))
    
    # Sort by row ascending, then by column descending
    # For sorting by column descending, we use negative column index
    coordinates.sort(key=lambda coord: (coord[0], -coord[1]))
    
    return coordinates