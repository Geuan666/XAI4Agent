def do_algebra(operator, operand):
    """
    Given two lists operator, and operand. The first list has basic algebra operations, and 
    the second list is a list of integers. Use the two given lists to build the algebric 
    expression and return the evaluation of this expression.

    The basic algebra operations:
    Addition ( + ) 
    Subtraction ( - ) 
    Multiplication ( * ) 
    Floor division ( // ) 
    Exponentiation ( ** ) 

    Example:
    operator['+', '*', '-']
    array = [2, 3, 4, 5]
    result = 2 + 3 * 4 - 5
    => result = 9

    Note:
        The length of operator list is equal to the length of operand list minus one.
        Operand is a list of of non-negative integers.
        Operator list has at least one operator, and operand list has at least two operands.

    """
    # Create a copy of the operand list to avoid modifying the original
    values = operand[:]
    
    # First, handle exponentiation (**)
    i = 0
    while i < len(operator):
        if operator[i] == '**':
            values[i] = values[i] ** values[i+1]
            values.pop(i+1)
            operator.pop(i)
        else:
            i += 1
            
    # Then, handle multiplication and floor division (* and //)
    i = 0
    while i < len(operator):
        if operator[i] == '*':
            values[i] = values[i] * values[i+1]
            values.pop(i+1)
            operator.pop(i)
        elif operator[i] == '//':
            values[i] = values[i] // values[i+1]
            values.pop(i+1)
            operator.pop(i)
        else:
            i += 1
            
    # Finally, handle addition and subtraction (+ and -)
    i = 0
    while i < len(operator):
        if operator[i] == '+':
            values[i] = values[i] + values[i+1]
            values.pop(i+1)
            operator.pop(i)
        elif operator[i] == '-':
            values[i] = values[i] - values[i+1]
            values.pop(i+1)
            operator.pop(i)
        else:
            i += 1
            
    return values[0]

# Test the function
if __name__ == "__main__":
    # Test case from the docstring
    op = ['+', '*', '-']
    arr = [2, 3, 4, 5]
    result = do_algebra(op, arr)
    print(f"Result: {result}")
    print("Expected: 9")
    
    # Additional test cases
    # Test with exponentiation
    op2 = ['**', '+']
    arr2 = [2, 3, 4]
    result2 = do_algebra(op2, arr2)
    print(f"Result2: {result2}")
    print("Expected: 12 (2^3 + 4 = 8 + 4)")
    
    # Test with division
    op3 = ['//', '+']
    arr3 = [10, 2, 3]
    result3 = do_algebra(op3, arr3)
    print(f"Result3: {result3}")
    print("Expected: 8 (10//2 + 3 = 5 + 3)")