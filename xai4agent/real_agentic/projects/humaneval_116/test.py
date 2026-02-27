def sort_array(arr):
    """
    In this Kata, you have to sort an array of non-negative integers according to
    number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    >>> sort_array([1, 5, 2, 3, 4])
    [1, 2, 4, 3, 5]
    >>> sort_array([0, 1, 2, 3, 4])
    [0, 1, 2, 4, 3]
    """
    # Sort by number of 1s in binary representation, then by decimal value
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))

# Test cases
if __name__ == "__main__":
    # Test case 1: [1, 5, 2, 3, 4]
    # Binary representations: 1->1, 5->101, 2->10, 3->11, 4->100
    # Ones count: 1->1, 5->2, 2->1, 3->2, 4->1
    # Sorted by ones count then decimal value: [1,2,4,3,5]
    result1 = sort_array([1, 5, 2, 3, 4])
    print(f"Test 1: {result1}")
    
    # Test case 2: [0, 1, 2, 3, 4] 
    # Binary representations: 0->0, 1->1, 2->10, 3->11, 4->100
    # Ones count: 0->0, 1->1, 2->1, 3->2, 4->1
    # Sorted by ones count then decimal value: [0, 1, 2, 4, 3]
    result2 = sort_array([0, 1, 2, 3, 4])
    print(f"Test 2: {result2}")
    
    # Let's manually verify the sorting
    test_arr = [1, 5, 2, 3, 4]
    for num in test_arr:
        print(f"{num} -> binary: {bin(num)}, ones: {bin(num).count('1')}")