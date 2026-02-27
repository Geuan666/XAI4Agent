def check_dict_case(dict):
    """
    Given a dictionary, return True if all keys are strings in lower
    case or all keys are strings in upper case, else return False.
    The function should return False is the given dictionary is empty.
    Examples:
    check_dict_case({"a":"apple", "b":"banana"}) should return True.
    check_dict_case({"a":"apple", "A":"banana", "B":"banana"}) should return False.
    check_dict_case({"a":"apple", 8:"banana", "a":"apple"}) should return False.
    check_dict_case({"Name":"John", "Age":"36", "City":"Houston"}) should return False.
    check_dict_case({"STATE":"NC", "ZIP":"12345" }) should return True.
    """
    # 如果字典为空，返回 False
    if not dict:
        return False

    # 检查所有键是否都是字符串
    if not all(isinstance(key, str) for key in dict.keys()):
        return False

    # 获取所有键
    keys = list(dict.keys())

    # 检查是否所有键都是小写
    all_lower = all(key.islower() for key in keys)

    # 检查是否所有键都是大写
    all_upper = all(key.isupper() for key in keys)

    # 返回是否所有键都是小写或所有键都是大写
    return all_lower or all_upper
