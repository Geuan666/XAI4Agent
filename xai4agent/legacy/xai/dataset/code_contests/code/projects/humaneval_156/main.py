def solve():
    """
    1622_C. Set or Decrease
    
    You are given an integer array a_1, a_2, ..., a_n and integer k.
    
    In one step you can 
    
      * either choose some index i and decrease a_i by one (make a_i = a_i - 1); 
      * or choose two indices i and j and set a_i equal to a_j (make a_i = a_j). 
    
    
    
    What is the minimum number of steps you need to make the sum of array ∑_{i=1}^{n}{a_i} ≤ k? (You are allowed to make values of array negative).
    
    Input
    
    The first line contains a single integer t (1 ≤ t ≤ 10^4) — the number of test cases.
    
    The first line of each test case contains two integers n and k (1 ≤ n ≤ 2 ⋅ 10^5; 1 ≤ k ≤ 10^{15}) — the size of array a and upper bound on its sum.
    
    The second line of each test case contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^9) — the array itself.
    
    It's guaranteed that the sum of n over all test cases doesn't exceed 2 ⋅ 10^5.
    
    Output
    
    For each test case, print one integer — the minimum number of steps to make ∑_{i=1}^{n}{a_i} ≤ k.
    
    Example
    
    Input
    
    
    4
    1 10
    20
    2 69
    6 9
    7 8
    1 2 1 3 1 2 1
    10 1
    1 2 3 1 2 6 1 6 8 10
    
    
    Output
    
    
    10
    0
    2
    7
    
    Note
    
    In the first test case, you should decrease a_1 10 times to get the sum lower or equal to k = 10.
    
    In the second test case, the sum of array a is already less or equal to 69, so you don't need to change it.
    
    In the third test case, you can, for example: 
    
      1. set a_4 = a_3 = 1; 
      2. decrease a_4 by one, and get a_4 = 0. 
    
    As a result, you'll get array [1, 2, 1, 0, 1, 2, 1] with sum less or equal to 8 in 1 + 1 = 2 steps.
    
    In the fourth test case, you can, for example: 
    
      1. choose a_7 and decrease in by one 3 times; you'll get a_7 = -2; 
      2. choose 4 elements a_6, a_8, a_9 and a_{10} and them equal to a_7 = -2. 
    
    As a result, you'll get array [1, 2, 3, 1, 2, -2, -2, -2, -2, -2] with sum less or equal to 1 in 3 + 4 = 7 steps.
    """
    pass
