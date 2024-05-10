import re

# Dictionary mapping month numbers to Indonesian names
month_names = {
    '1': 'januari',
    '2': 'februari',
    '3': 'maret',
    '4': 'april',
    '5': 'mei',
    '6': 'juni',
    '7': 'juli',
    '8': 'agustus',
    '9': 'september',
    '10': 'oktober',
    '11': 'november',
    '12': 'desember'
}

# Input text
text = "This is a date (22/50) and another date (01/12)."

# Regular expression to find date patterns
date_pattern = r'\((\d{1,2})/(\d{1,2})\)'

# Function to replace date patterns with month names
def replace_date(match):
    day, month = match.groups()
    return f'{int(day)} {month_names[month]}'

# Replace date patterns in the text
result = re.sub(date_pattern, replace_date, text)

print(result)
