def minSubArraySum(nums):
    """
    Given an array of integers nums, find the minimum sum of any non-empty sub-array
    of nums.
    Example
    minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
    minSubArraySum([-1, -2, -3]) == -6
    """
    if not nums:
        return 0
    
    # Initialize variables
    min_ending_here = nums[0]
    min_so_far = nums[0]
    
    # Iterate through the rest of the array
    for i in range(1, len(nums)):
        # Either extend the existing subarray or start a new one
        min_ending_here = min(nums[i], min_ending_here + nums[i])
        # Update the overall minimum
        min_so_far = min(min_so_far, min_ending_here)
    
    return min_so_far