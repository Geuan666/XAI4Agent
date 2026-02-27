def solve():
    """
    1582_E. Pchelyonok and Segments
    
    Pchelyonok decided to give Mila a gift. Pchelenok has already bought an array a of length n, but gifting an array is too common. Instead of that, he decided to gift Mila the segments of that array!
    
    Pchelyonok wants his gift to be beautiful, so he decided to choose k non-overlapping segments of the array [l_1,r_1], [l_2,r_2], … [l_k,r_k] such that:
    
      * the length of the first segment [l_1,r_1] is k, the length of the second segment [l_2,r_2] is k-1, …, the length of the k-th segment [l_k,r_k] is 1 
      * for each i<j, the i-th segment occurs in the array earlier than the j-th (i.e. r_i<l_j) 
      * the sums in these segments are strictly increasing (i.e. let sum(l … r) = ∑_{i=l}^{r} a_i — the sum of numbers in the segment [l,r] of the array, then sum(l_1 … r_1) < sum(l_2 … r_2) < … < sum(l_k … r_k)). 
    
    
    
    Pchelenok also wants his gift to be as beautiful as possible, so he asks you to find the maximal value of k such that he can give Mila a gift!
    
    Input
    
    The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases. The next 2 ⋅ t lines contain the descriptions of test cases. The description of each test case consists of two lines.
    
    The first line of each test case contains a single integer n (1 ≤ n ≤ 10^5) — the length of the array.
    
    The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^9) — the elements of the array a.
    
    It is guaranteed that the sum of n over all test cases does not exceed 10^5.
    
    Output
    
    For each test case, print the maximum possible value of k.
    
    Example
    
    Input
    
    
    5
    1
    1
    3
    1 2 3
    5
    1 1 2 2 3
    7
    1 2 1 1 3 2 6
    5
    9 6 7 9 7
    
    
    Output
    
    
    1
    1
    2
    3
    1
    """
    pass
