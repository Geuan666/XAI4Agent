def valid_date(date):
    """You have to write a function which validates a given date string and
    returns True if the date is valid otherwise False.
    The date is valid if all of the following rules are satisfied:
    1. The date string is not empty.
    2. The number of days is not less than 1 or higher than 31 days for months 1,3,5,7,8,10,12. And the number of days is not less than 1 or higher than 30 days for months 4,6,9,11. And, the number of days is not less than 1 or higher than 29 for the month 2.
    3. The months should not be less than 1 or higher than 12.
    4. The date should be in the format: mm-dd-yyyy

    for example:
    valid_date('03-11-2000') => True

    valid_date('15-01-2012') => False

    valid_date('04-0-2040') => False

    valid_date('06-04-2020') => True

    valid_date('06/04/2020') => False
    """
    # Check if date string is empty
    if not date:
        return False

    # Check if format is mm-dd-yyyy
    parts = date.split('-')
    if len(parts) != 3:
        return False

    try:
        month_str, day_str, year_str = parts
        # Check if all parts are numeric and have correct length
        if not (month_str.isdigit() and day_str.isdigit() and year_str.isdigit()):
            return False

        if len(month_str) != 2 or len(day_str) != 2 or len(year_str) != 4:
            return False

        month = int(month_str)
        day = int(day_str)
        year = int(year_str)

        # Check month range
        if month < 1 or month > 12:
            return False

        # Days in each month (non-leap year)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Adjust for leap year if needed (though not required for this problem)
        # For simplicity, we'll use the fixed values from the docstring

        # Check day range based on month
        max_days = days_in_month[month - 1]
        if month == 2:  # February
            max_days = 29  # According to docstring, Feb can have up to 29 days

        if day < 1 or day > max_days:
            return False

        return True
    except ValueError:
        return False
