def solve():
    """
    1599_I. Desert
    
    You are given an undirected graph of N nodes and M edges, E_1, E_2, ... E_M.
    
    A connected graph is a cactus if each of it's edges belogs to at most one simple cycle. A graph is a desert if each of it's connected components is a cactus. 
    
    Find the number of pairs (L, R), (1 ≤ L ≤ R ≤ M) such that, if we delete all the edges except for E_L, E_{L+1}, ... E_R, the graph is a desert. 
    
    Input
    
    The first line contains two integers N and M (2 ≤ N ≤ 2.5 × 10^5, 1 ≤ M ≤ 5 × 10^5). Each of the next M lines contains two integers. The i-th line describes the i-th edge. It contains integers U_i and V_i, the nodes connected by the i-th edge (E_i=(U_i, V_i)). It is guaranteed that 1 ≤ U_i, V_i ≤ N and U_i ≠ V_i. 
    
    Output
    
    The output contains one integer number – the answer.
    
    Examples
    
    Input
    
    
    5 6
    1 2
    2 3
    3 4
    4 5
    5 1
    2 4
    
    
    Output
    
    
    20
    
    Input
    
    
    2 3
    1 2
    1 2
    1 2
    
    
    Output
    
    
    5
    
    Note
    
    In the second example: Graphs for pairs (1, 1), (2, 2) and (3, 3) are deserts because they don't have any cycles. Graphs for pairs (1, 2) and (2, 3) have one cycle of length 2 so they are deserts.
    """
    pass
