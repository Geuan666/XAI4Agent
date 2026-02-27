def strange_sort_list(lst):
    '''
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Examples:
    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
    strange_sort_list([]) == []
    '''
    if not lst:
        return []

    # 创建已排序的副本以避免修改原列表
    sorted_lst = sorted(lst)
    result = []
    left, right = 0, len(sorted_lst) - 1

    # 交替选择最小值和最大值
    while left <= right:
        # 选择最小值
        result.append(sorted_lst[left])
        left += 1

        # 如果还有元素，选择最大值
        if left <= right:
            result.append(sorted_lst[right])
            right -= 1

    return result
