def solve():
    """
    1575_J. Jeopardy of Dropped Balls
    
    Mr. Chanek has a new game called Dropping Balls. Initially, Mr. Chanek has a grid a of size n × m
    
    Each cell (x,y) contains an integer a_{x,y} denoting the direction of how the ball will move.
    
      * a_{x,y}=1 — the ball will move to the right (the next cell is (x, y + 1)); 
      * a_{x,y}=2 — the ball will move to the bottom (the next cell is (x + 1, y)); 
      * a_{x,y}=3 — the ball will move to the left (the next cell is (x, y - 1)). 
    
    
    
    Every time a ball leaves a cell (x,y), the integer a_{x,y} will change to 2. Mr. Chanek will drop k balls sequentially, each starting from the first row, and on the c_1, c_2, ..., c_k-th (1 ≤ c_i ≤ m) columns.
    
    Determine in which column each ball will end up in (position of the ball after leaving the grid).
    
    Input
    
    The first line contains three integers n, m, and k (1 ≤ n, m ≤ 1000, 1 ≤ k ≤ 10^5) — the size of the grid and the number of balls dropped by Mr. Chanek.
    
    The i-th of the next n lines contains m integers a_{i,1},a_{i,2},…,a_{i,m} (1 ≤ a_{i,j} ≤ 3). It will satisfy a_{i, 1} ≠ 3 and a_{i, m} ≠ 1.
    
    The next line contains k integers c_1, c_2, …, c_k (1 ≤ c_i ≤ m) — the balls' column positions dropped by Mr. Chanek sequentially.
    
    Output
    
    Output k integers — the i-th integer denoting the column where the i-th ball will end.
    
    Examples
    
    Input
    
    
    5 5 3
    1 2 3 3 3
    2 2 2 2 2
    2 2 2 2 2
    2 2 2 2 2
    2 2 2 2 2
    1 2 1
    
    
    Output
    
    
    2 2 1 
    
    
    Input
    
    
    1 2 2
    1 3
    1 2
    
    
    Output
    
    
    1 2 
    
    Note
    
    In the first example, the first ball will drop as follows. Note that the cell (1, 1) will change direction to the bottom direction.
    
    <image>
    
    The second and third balls will drop as follows. 
    
    <image>
    
    All balls will be dropped from the first row and on the c_1, c_2, ..., c_k-th columns respectively. A ball will stop dropping once it leaves the grid.
    """
    pass
