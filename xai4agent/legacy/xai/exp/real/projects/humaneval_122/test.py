def add_elements(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return
    the sum of the elements with at most two digits from the first k elements of arr.

    Example:

        Input: arr = [111,21,3,4000,5,6,7,8,9], k = 4
        Output: 24 # sum of 21 + 3

    Constraints:
        1. 1 <= len(arr) <= 100
        2. 1 <= k <= len(arr)
    """
    total = 0
    for i in range(min(k, len(arr))):
        if 0 <= arr[i] <= 99:
            total += arr[i]
    return total

# Test with the provided example
arr = [111,21,3,4000,5,6,7,8,9]
k = 4
result = add_elements(arr, k)
print(f"Input: arr = {arr}, k = {k}")
print(f"Output: {result}")

# Additional test cases
print("Additional test cases:")
print(f"add_elements([1,2,3,4,5], 3) = {add_elements([1,2,3,4,5], 3)}")
print(f"add_elements([100,200,3], 2) = {add_elements([100,200,3], 2)}")