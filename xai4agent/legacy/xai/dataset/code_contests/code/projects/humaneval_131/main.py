def solve():
    """
    1617_E. Christmas Chocolates
    
    Christmas is coming, Icy has just received a box of chocolates from her grandparents! The box contains n chocolates. The i-th chocolate has a non-negative integer type a_i.
    
    Icy believes that good things come in pairs. Unfortunately, all types of chocolates are distinct (all a_i are distinct). Icy wants to make at least one pair of chocolates the same type. 
    
    As a result, she asks her grandparents to perform some chocolate exchanges. Before performing any chocolate exchanges, Icy chooses two chocolates with indices x and y (1 ≤ x, y ≤ n, x ≠ y).
    
    In a chocolate exchange, Icy's grandparents choose a non-negative integer k, such that 2^k ≥ a_x, and change the type of the chocolate x from a_x to 2^k - a_x (that is, perform a_x := 2^k - a_x).
    
    The chocolate exchanges will be stopped only when a_x = a_y. Note that other pairs of equal chocolate types do not stop the procedure.
    
    Icy's grandparents are smart, so they would choose the sequence of chocolate exchanges that minimizes the number of exchanges needed. Since Icy likes causing trouble, she wants to maximize the minimum number of exchanges needed by choosing x and y appropriately. She wonders what is the optimal pair (x, y) such that the minimum number of exchanges needed is maximized across all possible choices of (x, y).
    
    Since Icy is not good at math, she hopes that you can help her solve the problem.
    
    Input
    
    The first line of the input contains a single integer n (2 ≤ n ≤ 2 ⋅ 10^5) — the number of chocolates.
    
    The second line of the input contains n integers a_1, a_2, ..., a_n (0 ≤ a_i ≤ 10^9).
    
    It is guaranteed that all a_i are distinct.
    
    Output
    
    Output three integers x, y, and m.
    
    x and y are indices of the optimal chocolates to perform exchanges on. Your output must satisfy 1 ≤ x, y ≤ n, x ≠ y.
    
    m is the number of exchanges needed to obtain a_x = a_y. We can show that m ≤ 10^9 for any pair of chocolates.
    
    If there are multiple solutions, output any.
    
    Examples
    
    Input
    
    
    5
    5 6 7 8 9
    
    
    Output
    
    
    2 5 5
    
    
    Input
    
    
    2
    4 8
    
    
    Output
    
    
    1 2 2
    
    Note
    
    In the first test case, the minimum number of exchanges needed to exchange a chocolate of type 6 to a chocolate of type 9 is 5. The sequence of exchanges is as follows: 6 → 2 → 0 → 1 → 7 → 9.
    
    In the second test case, the minimum number of exchanges needed to exchange a chocolate of type 4 to a chocolate of type 8 is 2. The sequence of exchanges is as follows: 4 → 0 → 8.
    """
    pass
