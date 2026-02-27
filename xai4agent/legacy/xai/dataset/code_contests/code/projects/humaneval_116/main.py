def solve():
    """
    1613_D. MEX Sequences
    
    Let's call a sequence of integers x_1, x_2, ..., x_k MEX-correct if for all i (1 ≤ i ≤ k) |x_i - \operatorname{MEX}(x_1, x_2, ..., x_i)| ≤ 1 holds. Where \operatorname{MEX}(x_1, ..., x_k) is the minimum non-negative integer that doesn't belong to the set x_1, ..., x_k. For example, \operatorname{MEX}(1, 0, 1, 3) = 2 and \operatorname{MEX}(2, 1, 5) = 0.
    
    You are given an array a consisting of n non-negative integers. Calculate the number of non-empty MEX-correct subsequences of a given array. The number of subsequences can be very large, so print it modulo 998244353. 
    
    Note: a subsequence of an array a is a sequence [a_{i_1}, a_{i_2}, ..., a_{i_m}] meeting the constraints 1 ≤ i_1 < i_2 < ... < i_m ≤ n. If two different ways to choose the sequence of indices [i_1, i_2, ..., i_m] yield the same subsequence, the resulting subsequence should be counted twice (i. e. two subsequences are different if their sequences of indices [i_1, i_2, ..., i_m] are not the same).
    
    Input
    
    The first line contains a single integer t (1 ≤ t ≤ 10^5) — the number of test cases.
    
    The first line of each test case contains a single integer n (1 ≤ n ≤ 5 ⋅ 10^5).
    
    The second line contains n integers a_1, a_2, ..., a_n (0 ≤ a_i ≤ n).
    
    The sum of n over all test cases doesn't exceed 5 ⋅ 10^5.
    
    Output
    
    For each test case, print a single integer — the number of non-empty MEX-correct subsequences of a given array, taken modulo 998244353.
    
    Example
    
    Input
    
    
    4
    3
    0 2 1
    2
    1 0
    5
    0 0 0 0 0
    4
    0 1 2 3
    
    
    Output
    
    
    4
    2
    31
    7
    
    Note
    
    In the first example, the valid subsequences are [0], [1], [0,1] and [0,2].
    
    In the second example, the valid subsequences are [0] and [1].
    
    In the third example, any non-empty subsequence is valid.
    """
    pass
