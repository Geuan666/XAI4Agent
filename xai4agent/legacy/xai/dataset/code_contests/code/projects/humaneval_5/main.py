def solve():
    """
    1575_F. Finding Expected Value
    
    Mr. Chanek opened a letter from his fellow, who is currently studying at Singanesia. Here is what it says.
    
    Define an array b (0 ≤ b_i < k) with n integers. While there exists a pair (i, j) such that b_i ≠ b_j, do the following operation:
    
      * Randomly pick a number i satisfying 0 ≤ i < n. Note that each number i has a probability of 1/n to be picked. 
      * Randomly Pick a number j satisfying 0 ≤ j < k. 
      * Change the value of b_i to j. It is possible for b_i to be changed to the same value. 
    
    
    
    Denote f(b) as the expected number of operations done to b until all elements of b are equal. 
    
    You are given two integers n and k, and an array a (-1 ≤ a_i < k) of n integers. 
    
    For every index i with a_i = -1, replace a_i with a random number j satisfying 0 ≤ j < k. Let c be the number of occurrences of -1 in a. There are k^c possibilites of a after the replacement, each with equal probability of being the final array.
    
    Find the expected value of f(a) modulo 10^9 + 7. 
    
    Formally, let M = 10^9 + 7. It can be shown that the answer can be expressed as an irreducible fraction p/q, where p and q are integers and q not ≡ 0 \pmod{M}. Output the integer equal to p ⋅ q^{-1} mod M. In other words, output such an integer x that 0 ≤ x < M and x ⋅ q ≡ p \pmod{M}.
    
    After reading the letter, Mr. Chanek gave the task to you. Solve it for the sake of their friendship!
    
    Input
    
    The first line contains two integers n and k (2 ≤ n ≤ 10^5, 2 ≤ k ≤ 10^9). 
    
    The second line contains n integers a_1, a_2, …, a_n (-1 ≤ a_i < k).
    
    Output
    
    Output an integer denoting the expected value of f(a) modulo 10^9 + 7.
    
    Examples
    
    Input
    
    
    2 2
    0 1
    
    
    Output
    
    
    2
    
    
    Input
    
    
    2 2
    0 -1
    
    
    Output
    
    
    1
    
    
    Input
    
    
    3 3
    0 1 1
    
    
    Output
    
    
    12
    
    
    Input
    
    
    3 3
    -1 -1 -1
    
    
    Output
    
    
    11
    
    
    Input
    
    
    10 9
    -1 0 -1 1 1 2 2 3 3 3
    
    
    Output
    
    
    652419213
    """
    pass
