# 二分查找
def find_largest_index_less_than_target(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result

