METADATA = {}


def check(candidate):
    assert candidate([1, 2, 4, 10]) == True
    assert candidate([1, 2, 4, 20]) == True
    assert candidate([1, 20, 4, 10]) == False
    assert candidate([4, 1, 0, -10]) == True
    assert candidate([4, 1, 1, 0]) == True
    assert candidate([1, 2, 3, 2, 5, 60]) == False
    assert candidate([1, 2, 3, 4, 5, 60]) == True
    assert candidate([9, 9, 9, 9]) == True

if __name__ == "__main__":
    import sys
    import importlib.util
    import traceback

    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    func_name = "monotonic"

    if hasattr(main_module, func_name):
        func = getattr(main_module, func_name)
        try:
            check(func)
            print("\n✓ SUCCESS: All tests passed!")
        except AssertionError:
            print("\n✗ FAILURE: Test failed")
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Error: Function '{func_name}' not found in main.py")
        sys.exit(1)

