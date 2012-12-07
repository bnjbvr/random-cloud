"""
    unit_test_category_splits.py
    Author: Benjamin Bouvier

    Prints all possible category splits for a given category.
"""
from RandomForest import *

if __name__ == '__main__':
    possible_values = ['A', 'B', 'C', 'D']
    splits = generate_category_choice( possible_values )
    for s in splits:
        print "Split: ", s.feature_range
