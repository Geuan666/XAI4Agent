def minPath(grid, k):
    """
    Given a grid with N rows and N columns (N >= 2) and a positive integer k, 
    each cell of the grid contains a value. Every integer in the range [1, N * N]
    inclusive appears exactly once on the cells of the grid.

    You have to find the minimum path of length k in the grid. You can start
    from any cell, and in each step you can move to any of the neighbor cells,
    in other words, you can go to cells which share an edge with you current
    cell.
    Please note that a path of length k means visiting exactly k cells (not
    necessarily distinct).
    You CANNOT go off the grid.
    A path A (of length k) is considered less than a path B (of length k) if
    after making the ordered lists of the values on the cells that A and B go
    through (let's call them lst_A and lst_B), lst_A is lexicographically less
    than lst_B, in other words, there exist an integer index i (1 <= i <= k)
    such that lst_A[i] < lst_B[i] and for any j (1 <= j < i) we have
    lst_A[j] = lst_B[j].
    It is guaranteed that the answer is unique.
    Return an ordered list of the values on the cells that the minimum path go through.

    Examples:

        Input: grid = [ [1,2,3], [4,5,6], [7,8,9]], k = 3
        Output: [1, 2, 1]

        Input: grid = [ [5,9,3], [4,1,6], [7,8,2]], k = 1
        Output: [1]
    """
    n = len(grid)
    
    # For k=1, return the minimum value in the grid
    if k == 1:
        min_val = float('inf')
        for i in range(n):
            for j in range(n):
                min_val = min(min_val, grid[i][j])
        return [min_val]
    
    # Directions for moving to neighbors
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Find all possible paths of length k and return the lexicographically smallest
    def dfs(row, col, path, remaining):
        if remaining == 0:
            return [path[:]]
        
        result = []
        # Get all valid neighbors
        neighbors = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < n and 0 <= new_col < n:
                neighbors.append((grid[new_row][new_col], new_row, new_col))
        
        # Sort neighbors by value to try smaller values first
        neighbors.sort()
        
        for val, new_row, new_col in neighbors:
            path.append(val)
            result.extend(dfs(new_row, new_col, path, remaining - 1))
            path.pop()
            
        return result
    
    # Try starting from each cell
    all_paths = []
    for i in range(n):
        for j in range(n):
            path = [grid[i][j]]
            all_paths.extend(dfs(i, j, path, k - 1))
    
    # Return the lexicographically smallest path
    if all_paths:
        all_paths.sort()
        return all_paths[0]
    else:
        return []