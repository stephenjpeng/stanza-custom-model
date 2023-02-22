# Task: transform the weight and date to a list of possible formats
# 200 -> [200, “two hundred”]
# 2/23/2023 -> [“2/23/2023”, “February 23rd, 2023”, “02-23-2023”]

# python3 transform_weight_date.py 100 2017-04-08

from num2words import num2words
from datetime import datetime
import sys

def number_to_words(num):
    """Transform num into word
    Args:
        num (string or int): the number that we want to transform. e.g. 200

    Returns:
        [num, num_words]: return a list of num (string type) 
                        and num_words (the word in english of that number)
    """
    if isinstance(num, int):
        num = str(num)
    num_words = num2words(num)
    return [num, num_words]


def date_to_formats(date_str):
    """Transform the date into a list of possible date formats

    Args:
        date_str (string): the date that we want to transform. e.g. "2017-04-07"

    Returns:
        result: a list of possible date formats
    """
    formats = ["%B %e, %Y", "%e %b %Y", "%x", "%m/%d/%Y", "%m-%d-%Y", "%e %B %Y"]
    
    # Based on generated data, the date has the format: 2017-04-07
    date_ = datetime.strptime(date_str, "%Y-%m-%d")
    result = []
    for fmt in formats:
        result.append(date_.strftime(fmt))
    result.insert(0, date_str)
    print(result)
    return result

if __name__ == "__main__":
    num = sys.argv[1]
    num_list = number_to_words(num)
    
    date = sys.argv[2]
    date_list = date_to_formats(date)
    
    print(num_list)
    print(date_list)