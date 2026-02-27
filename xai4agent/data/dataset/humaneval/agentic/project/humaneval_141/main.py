def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns
    'Yes' if the the file's name is valid, and returns 'No' otherwise.
    A file's name is considered to be valid if and only if all the following conditions
    are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'
    - The substring before the dot should not be empty, and it starts with a letter from
    the latin alphapet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check("example.txt") # => 'Yes'
    file_name_check("1example.dll") # => 'No' (the name should start with a latin alphapet letter)
    """
    # 检查是否有且仅有一个点
    if file_name.count('.') != 1:
        return 'No'

    # 分割文件名
    name_part, ext_part = file_name.split('.')

    # 检查扩展名
    if ext_part not in ['txt', 'exe', 'dll']:
        return 'No'

    # 检查文件名部分是否为空
    if not name_part:
        return 'No'

    # 检查文件名是否以字母开头
    if not name_part[0].isalpha():
        return 'No'

    # 统计数字个数
    digit_count = sum(1 for c in file_name if c.isdigit())
    if digit_count > 3:
        return 'No'

    return 'Yes'
