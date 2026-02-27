def solve():
    """
    1603_D. Artistic Partition
    
    For two positive integers l and r (l ≤ r) let c(l, r) denote the number of integer pairs (i, j) such that l ≤ i ≤ j ≤ r and \operatorname{gcd}(i, j) ≥ l. Here, \operatorname{gcd}(i, j) is the [greatest common divisor (GCD)](https://en.wikipedia.org/wiki/Greatest_common_divisor) of integers i and j.
    
    YouKn0wWho has two integers n and k where 1 ≤ k ≤ n. Let f(n, k) denote the minimum of ∑_{i=1}^{k}{c(x_i+1,x_{i+1})} over all integer sequences 0=x_1 < x_2 < … < x_{k} < x_{k+1}=n.
    
    Help YouKn0wWho find f(n, k). 
    
    Input
    
    The first line contains a single integer t (1 ≤ t ≤ 3 ⋅ 10^5) — the number of test cases.
    
    The first and only line of each test case contains two integers n and k (1 ≤ k ≤ n ≤ 10^5). 
    
    Output
    
    For each test case, print a single integer — f(n, k).
    
    Example
    
    Input
    
    
    4
    6 2
    4 4
    3 1
    10 3
    
    
    Output
    
    
    8
    4
    6
    11
    
    Note
    
    In the first test case, YouKn0wWho can select the sequence [0, 2, 6]. So f(6, 2) = c(1, 2) + c(3, 6) = 3 + 5 = 8 which is the minimum possible.
    """
    pass
