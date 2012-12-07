"""
    RandomForest.py
    Author: Benjamin Bouvier

    This file contains all the functions necessary to run the RandomForest algorithm
    from scratch.
"""

import csv
import math
from random import randint

from Record import *
from Dataset import *
from DecisionTree import *
from Split import *

def grow_forest( n, records ):
    """Grow a forest of n trees, using the given records.
    Returns a list containing all the trees (the "forest")."""
    dataset = Dataset( records )
    record_number = dataset.size

    dts = []
    for i in xrange(n):
        print "Training", i
        # pick randomly as many records as the number in the dataset.
        picked_records = []
        for j in xrange( record_number ):
            ind_picked = randint(0, record_number-1)
            picked_records.append( dataset[ ind_picked ] )
        picked_records = Dataset( picked_records )
        # train a tree with these records and add it to the forest
        tree = train(picked_records)
        dts.append( tree )
    return dts

features_type = {}
def feature_is_numerical(records, index):
    """Returns true if all values of the features at the given index are numerical in the
    given set of records, false otherwise.
    This allows to determine if we have to use the category or the numerical split.

    The features types are cached to avoid intense recomputation."""
    if index not in features_type:
        features_type[index] = True
        for r in records:
            try:
                a = float(r.features[index])
            except:
                features_type[index] = False
                return False
    return features_type[index]

def generate_category_choice(possible):
    """Generates all distinct category splits.
    possible: Possible values for the category.

    If there are n categories, there are 2^(n-1)-1 distinct possible splits.
    All the splits are generated and returned in a list.

    Uses the binary form of number from 1 to 2^(n-1)-1 to generate
    the splits."""
    n = len(possible)
    splits = []
    for i in range(1, pow(2, n-1)):
        split = Split(is_numerical=False)
        for j in xrange(n):
            if (i >> j) % 2 == 1:
                split.add_category_range( possible[j] )
        splits.append( split )
    return splits

def generate_category_splits( records, index ):
    """Generates and fill the category splits with the records.

    records is the dataset, index is the index of the category feature.
    returns a list containing all splits filled with records.
    """
    possible = {}
    for r in records:
        possible[ r.features[index] ] = True
    possible = possible.keys()
    splits = []
    for choice in generate_category_choice( possible ):
        choice.place( records, index )
        splits.append( choice )
    return splits

def generate_numerical_splits( records, index ):
    """Generates and fills all the possible numerical splits with the
    records, with respect to their feature in the given index.

    Returns the list of possible splits
    """
    possible = {}
    for r in records:
        possible[ r.features[index] ] = True
    possible = possible.keys()
    splits = []

    for i in xrange(0, len(possible)-1):
        s = Split(is_numerical=True)
        s.set_numerical_range( possible[i] )
        s.place( records, index )
        splits.append( s )

    return splits

def generate_splits( records, index ):
    """Generates the splits and returns them, using the given records
    and the feature at the given index."""
    splits = []
    is_numerical = feature_is_numerical( records, index )
    if is_numerical:
        splits = generate_numerical_splits( records, index )
    else:
        splits = generate_category_splits( records, index )

    return splits

def train(records):
    """Train a tree using the records.

    We randomly pick square_root(m) attributes each time we want
    to select some attributes.
    The maximal depth authorized is 1e30, which should be enough in
    most cases.
    """
    m = len( records[0].features )
    sqm = int(math.sqrt(m))
    return train_r( records, range(m), sqm, 1e30 )

def train_r(records, attributes, sqm, depth):
    """Recursive call of the train function.

    Use the given records and train a tree of maximal given depth,
    using sqm attributes among all the possible given attributes.

    The recursion stops when all records are the same, or have the same
    label, or the maximal depth is reached."""

    if records.label_monotone or records.monotone or depth == 0:
        return Decision(records.mode)

    chosen_attributes = []
    attributes_with_no_split = 0

    # this loop ensure that we select attributes with distinct values.
    while len(chosen_attributes) == attributes_with_no_split:
        # select randomly sqm elements
        chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]
        # repeat selection as long as at least one feature appears twice
        while len(list(set(chosen_attributes))) != len(chosen_attributes):
            chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]

        best_gain = -1
        former_best = None
        best_split = None

        best_index = None
        best_range = None
        is_numerical = None

        attributes_with_no_split = 0
        for criteria in chosen_attributes:
            splits = generate_splits( records, criteria )
            if len(splits) == 0:
                # there is no splits when all values of the feature
                # are the same
                attributes_with_no_split += 1

            for s in splits:
                gain = s.gain
                if best_gain < gain:
                    former_best = best_split
                    best_split = s
                    best_gain = gain
                    if former_best is not None:
                        del former_best
            del splits

    s = best_split
    decision_tree = DecisionTree( s.feature_index, s.feature_range, s.is_numerical )

    if s.left.size == 0 or s.right.size == 0:
        del s
        return Decision( records.mode )

    depth -= 1
    decision_tree.right = train_r( s.right, attributes, sqm, depth )
    decision_tree.left  = train_r( s.left, attributes, sqm, depth )
    del s

    return decision_tree

def major_vote( dts, record ):
    """Given a list of decision trees dts and a single record,
    returns the label with the highest votes.

    Used in tests only."""
    votes = {}
    for dt in dts:
        p = dt.vote( record )
        votes[ p ] = votes.get(p, 0) + 1
    best, max_vote = None, 0
    for k in votes:
        if votes[k] > max_vote:
            max_vote = votes[k]
            best = k
    return best
