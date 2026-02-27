def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1.
        Return the number of triples (a[i], a[j], a[k]) of a where i < j < k,
    and a[i] + a[j] + a[k] is a multiple of 3.

    Example :
        Input: n = 5
        Output: 1
        Explanation:
        a = [1, 3, 7, 13, 21]
        The only valid triple is (1, 7, 13).
    """
    # 统计模3余0和余1的元素个数
    count_mod0 = 0
    count_mod1 = 0

    # 根据公式 a[i] = i * i - i + 1 计算每个元素模3的结果
    for i in range(1, n + 1):
        val = i * i - i + 1
        if val % 3 == 0:
            count_mod0 += 1
        else:
            count_mod1 += 1

    # 三个数之和能被3整除的情况：
    # 1. 三个数都模3余0: C(count_mod0, 3)
    # 2. 三个数都模3余1: C(count_mod1, 3)
    # 3. 一个数模3余0，一个数模3余1，一个数模3余2（但这里没有模3余2的数）

    # 实际上，我们只需要考虑模3余0和模3余1的数
    # 因为只有这两种情况，而且我们不能有模3余2的数
    # 所以我们只能选3个模3余0的数或者3个模3余1的数

    # 计算组合数
    def combination(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        # 使用更高效的方式计算组合数
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result

    # 三个数都模3余0的组合数
    result = combination(count_mod0, 3)
    # 三个数都模3余1的组合数
    result += combination(count_mod1, 3)

    return result
