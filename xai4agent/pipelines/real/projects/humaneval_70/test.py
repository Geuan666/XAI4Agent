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
    
    # Create a copy and sort it to easily get min/max
    sorted_lst = sorted(lst)
    result = []
    take_min = True
    
    while len(sorted_lst) > 0:
        if take_min:
            # Take minimum (first element)
            result.append(sorted_lst.pop(0))
        else:
            # Take maximum (last element)
            result.append(sorted_lst.pop())
        take_min = not take_min
        
    return result

# Test cases
print(strange_sort_list([1, 2, 3, 4]))  # Expected: [1, 4, 2, 3]
print(strange_sort_list([5, 5, 5, 5]))  # Expected: [5, 5, 5, 5]
print(strange_sort_list([]))            # Expected: []