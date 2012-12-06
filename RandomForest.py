import csv
import math
from random import randint

#import cProfile # for profiling purposes only
#import guppy

from Record import *
from Dataset import *
from DecisionTree import *
from Split import *

def grow_forest( n, records, test_records = None ):
    dataset = Dataset( records )
    record_number = dataset.size

    dts = []
    for i in xrange(n):
        print "Training", i
        picked_records = []
        for j in xrange( record_number ):
            ind_picked = randint(0, record_number-1)
            dataset[ ind_picked ].pick()
            picked_records.append( dataset[ ind_picked ] )
        picked_records = Dataset( picked_records )
        tree = train(picked_records)
        dts.append( tree )
    return dts

features_type = {}
def feature_is_numerical(records, index):
    """Returns true if all values of the features at the given index are numerical in the
    given set of records, false otherwise."""
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
    """Generates all distinct category splits."""
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
    possible = {}
    for r in records:
        possible[ r.features[index] ] = True
    possible = possible.keys()
    splits = []

    for i in xrange(0, len(possible)-1):
        s = Split(is_numerical=True)
        s.set_numerical_range( i )
        s.place( records, index )
        splits.append( s )

    return splits

def generate_splits( records, index ):
    splits = []
    is_numerical = feature_is_numerical( records, index )
    if is_numerical:
        splits = generate_numerical_splits( records, index )
    else:
        splits = generate_category_splits( records, index )

    return splits

def train(records):
    m = len( records[0].features )
    sqm = int(math.sqrt(m))
    return train_r( records, range(m), sqm, 1e30 )

def train_r(records, attributes, sqm, depth):

    if records.label_monotone or records.monotone or depth == 0:
        return Decision(records.mode)

    chosen_attributes = []
    attributes_with_no_split = 0

    while len(chosen_attributes) == attributes_with_no_split:
        chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]
        while len(list(set(chosen_attributes))) != len(chosen_attributes):
            chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]

        if len(attributes) == 0:
            return Decision( records.mode )

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
                attributes_with_no_split += 1

            for s in splits:
                gain = s.gain
                if best_gain < gain:
                    former_best = best_split
                    best_split = s
                    best_gain = gain
                    if former_best is not None:
                        del former_best

    s = best_split
    decision_tree = DecisionTree( s.feature_index, s.feature_range, s.is_numerical )

    if s.left.size == 0 or s.right.size == 0:
        return Decision( records.mode )

    depth -= 1
    decision_tree.right = train_r( s.right, attributes, sqm, depth )
    decision_tree.left  = train_r( s.left, attributes, sqm, depth )

    return decision_tree

def major_vote( dts, record ):
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
