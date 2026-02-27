def move_one_ball(arr):
    """We have an array 'arr' of N integers arr[1], arr[2], ..., arr[N].The
    numbers in the array will be randomly ordered. Your task is to determine if
    it is possible to get an array sorted in non-decreasing order by performing
    the following operation on the given array:
        You are allowed to perform right shift operation any number of times.

    One right shift operation means shifting all elements of the array by one
    position in the right direction. The last element of the array will be moved to
    the starting position in the array i.e. 0th index.

    If it is possible to obtain the sorted array by performing the above operation
    then return True else return False.
    If the given array is empty then return True.

    Note: The given list is guaranteed to have unique elements.

    For Example:

    move_one_ball([3, 4, 5, 1, 2])==>True
    Explanation: By performin 2 right shift operations, non-decreasing order can
                 be achieved for the given array.
    move_one_ball([3, 5, 4, 1, 2])==>False
    Explanation:It is not possible to get non-decreasing order for the given
                array by performing any number of right shift operations.

    """
    # 空数组或单元素数组可以直接返回True
    if len(arr) <= 1:
        return True

    # 找到所有下降点
    drop_points = []
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            drop_points.append(i)

    # 如果没有下降点，说明数组已经有序
    if len(drop_points) == 0:
        return True

    # 如果有超过一个下降点，无法通过右移得到有序数组
    if len(drop_points) > 1:
        return False

    # 如果只有一个下降点，检查是否可以通过右移得到有序数组
    # 下降点位置是drop_points[0]，意味着arr[drop_points[0]] > arr[drop_points[0]+1]
    # 如果将数组从drop_points[0]+1位置切分，前面部分和后面部分分别有序，
    # 并且后面部分的最大值不超过前面部分的最小值，则可以通过右移得到有序数组
    drop_index = drop_points[0]

    # 检查最后一个元素是否不大于第一个元素（这是右移后能形成有序数组的关键条件）
    return arr[-1] <= arr[0]
