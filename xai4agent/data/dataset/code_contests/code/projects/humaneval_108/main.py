def solve():
    """
    1608_C. Game Master
    
    n players are playing a game. 
    
    There are two different maps in the game. For each player, we know his strength on each map. When two players fight on a specific map, the player with higher strength on that map always wins. No two players have the same strength on the same map. 
    
    You are the game master and want to organize a tournament. There will be a total of n-1 battles. While there is more than one player in the tournament, choose any map and any two remaining players to fight on it. The player who loses will be eliminated from the tournament. 
    
    In the end, exactly one player will remain, and he is declared the winner of the tournament. For each player determine if he can win the tournament.
    
    Input
    
    The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases. The description of test cases follows.
    
    The first line of each test case contains a single integer n (1 ≤ n ≤ 10^5) — the number of players.
    
    The second line of each test case contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^9, a_i ≠ a_j for i ≠ j), where a_i is the strength of the i-th player on the first map. 
    
    The third line of each test case contains n integers b_1, b_2, ..., b_n (1 ≤ b_i ≤ 10^9, b_i ≠ b_j for i ≠ j), where b_i is the strength of the i-th player on the second map. 
    
    It is guaranteed that the sum of n over all test cases does not exceed 10^5.
    
    Output
    
    For each test case print a string of length n. i-th character should be "1" if the i-th player can win the tournament, or "0" otherwise.
    
    Example
    
    Input
    
    
    3
    4
    1 2 3 4
    1 2 3 4
    4
    11 12 20 21
    44 22 11 30
    1
    1000000000
    1000000000
    
    
    Output
    
    
    0001
    1111
    1
    
    Note
    
    In the first test case, the 4-th player will beat any other player on any game, so he will definitely win the tournament.
    
    In the second test case, everyone can be a winner. 
    
    In the third test case, there is only one player. Clearly, he will win the tournament.
    """
    pass
