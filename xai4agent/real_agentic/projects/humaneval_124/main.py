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
    # Rule 1: Check if date string is not empty
    if not date or len(date) == 0:
        return False
    
    # Rule 4: Check if date is in correct format (mm-dd-yyyy)
    parts = date.split('-')
    if len(parts) != 3:
        return False
    
    month_str, day_str, year_str = parts
    
    # Check if each part has correct length
    if len(month_str) != 2 or len(day_str) != 2 or len(year_str) != 4:
        return False
    
    # Check if all parts contain only digits
    if not (month_str.isdigit() and day_str.isdigit() and year_str.isdigit()):
        return False
    
    try:
        month = int(month_str)
        day = int(day_str)
        year = int(year_str)
    except ValueError:
        return False
    
    # Rule 3: Check if months are between 1-12
    if month < 1 or month > 12:
        return False
    
    # Rule 2: Check if days are valid for each month
    # Days in each month (non-leap year)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Adjust February for leap years
    if is_leap_year(year):
        days_in_month[1] = 29
    
    max_days = days_in_month[month - 1]
    
    if day < 1 or day > max_days:
        return False
    
    return True

def is_leap_year(year):
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)