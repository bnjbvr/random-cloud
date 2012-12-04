import csv
import cProfile # for profiling purposes only
import math
from random import randint

from Record import *
from Dataset import *
from DecisionTree import *
from Split import *

"""
from mrjob.job import MRJob

class MRWordCounter(MRJob):
    def mapper(self, key, line):
        for word in line.split():
            yield word, 1

    def reducer(self, word, occurrences):
        yield word, sum(occurrences)

    def steps(self):
        return [self.mr(self.mapper, self.reducer),]
"""

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
    #print "FEATURE", index, "NUMERICAL?", is_numerical
    if is_numerical:
        splits = generate_numerical_splits( records, index )
    else:
        splits = generate_category_splits( records, index )

    return splits

def train(records):
    m = len( records[0].features )
    sqm = int(math.sqrt(m))
    return train_r( records, range(m), sqm, 1e30)

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
                    best_split = s
                    best_gain = gain

    #print "best split obtained for attribute @ index", best_split.feature_index
    #print "with range ", best_split.feature_range

    s = best_split
    decision_tree = DecisionTree( s.feature_index, s.feature_range, s.is_numerical )

    if s.left.size == 0 or s.right.size == 0:
        return Decision( records.mode )

    depth -= 1
    decision_tree.right = train_r( s.right, attributes, sqm, depth )
    decision_tree.left  = train_r( s.left, attributes, sqm, depth )

    return decision_tree

def grow_forest( n, records ):
    dataset = Dataset( records )
    record_number = dataset.size        # N

    dts = []
    for i in xrange(n):
        print "Training", i
        #print "\n\nNEW TRAINING"
        picked_records = []
        for i in xrange( record_number ):
            ind_picked = randint(0, record_number-1)
            dataset[ ind_picked ].pick()
            picked_records.append( dataset[ ind_picked ] )
        picked_records = Dataset( picked_records )
        tree = train(picked_records)
        dts.append( tree )

        """
        print "\t> Training over. Validation..."

        test_records = []
        total, correct = 0, 0
        for r in records:
            if not r.picked:
                test_records.append( r )
        for r in test_records:
            if tree.vote(r) == r.label:
                correct += 1
            total += 1
        print "\t> Validation:", float(correct)/total

        for r in picked_records:
            r.picked = False
        """

    return dts

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

def main():
    records = []
    test_records = []
    i = 0
    #with open('examples.csv', 'r') as csvfile:
    #with open('titanic.csv', 'r') as csvfile:
    with open('train2.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if i < 1000:
                records.append( Record( row[:-1], row[-1] ) )
            else:
                test_records.append( Record( row[:-1], row[-1] ) )
            i += 1

    dts = grow_forest( 50, records )
    correct, total = 0, 0

    """
    for dt in dts:
        print 'dt:', dt
    """

    print "On records used for learning..."
    for r in records:
        total += 1
        if r.label == major_vote( dts, r ):
            correct += 1
        if total % 10 == 0:
            print "%s / %s correct (%s %)" % (correct, total, correct / float(total) * 100.)
    print "%s / %s correct (%s %)" % (correct, total, correct / float(total) * 100.)

    print "On test records..."
    total, correct = 0
    for r in test_records:
        total += 1
        if r.label == major_vote( dts, r ):
            correct += 1
        if total % 10 == 0:
            print "%s / %s correct (%s %)" % (correct, total, correct / float(total) * 100.)
    print "%s / %s correct (%s %)" % (correct, total, correct / float(total) * 100.)


if __name__ == '__main__':
    main()
    #cProfile.run('main()')
    #MRWordCounter.run()

def example():
    original_records = [ Record( ['>150', '1', 'Town', 'AT&T'], 'Yes' ),
            Record( ['<75', '1', 'Town', 'AT&T'], 'No' ),
            Record( ['<75', '2', 'City', 'Sprint'], 'No' ),
            Record( ['75..150', '2', 'City', 'MCI'], 'Yes' ),
            Record( ['75..150', '2', 'City', 'Sprint'], 'Yes' ),
            Record( ['75..150', '1', 'Town', 'MCI'], 'Yes' )]
    records = original_records[:]

    dt = train(records)

    print "\nEND OF TRAINING"
    print dt
    for r in records:
        print "Record", r
        p = dt.vote(r)
        print "Prediction:", p, '\n'
        if p != r.label:
            print "FATAL ERROR HERE"
            break

    """
    m = len(records[0].features)
    best_gain = None
    best_split = None

    best_index = None
    best_range = None
    is_numerical = None

    decision_tree = None

    for i in (1, 3):
        splits, isnum = generate_splits( records, i )
        for (meta_ind, meta_range, left, right) in splits:
            gain = gini_gain( records, left, right )
            print "SPLIT:\n%s\n%s" % ( left, right )
            print "Gain:", gain
            print "\n"
            if best_gain is None or best_gain < gain:
                best_split = (left, right)
                best_gain = gain
                best_index = meta_ind
                best_range = meta_range
                is_numerical = isnum
    print "Best split:", best_split[0], '\n', best_split[1]
    print "obtained for attribute @ index", best_index
    print "with range ", best_range

    decision_tree = DecisionTree( best_index, best_range, is_numerical )
    if gini(best_split[1]) == 0:
        print "Right subtree is pure"
        prediction = best_split[1][0].label
        print "Prediction would be:", prediction
        decision_tree.right = Decision( prediction )
    else:
# apply same procedure recursively to right tree
        pass

    records = left[:]
    best_gain = None
    for i in (0, 2):
        splits, isnum = generate_splits( records, i )
        for (meta_ind, meta_range, left, right) in splits:
            gain = gini_gain( records, left, right )
            print "SPLIT:\n%s\n%s" % ( left, right )
            print "Gain:", gain
            print "\n"
            if best_gain is None or best_gain < gain:
                best_split = (left, right)
                best_gain = gain
                best_index = meta_ind
                best_range = meta_range
    print "Best split:", best_split
    print "obtained for attribute @ index", best_index
    print "with range ", best_range

    decision_tree.left = DecisionTree( best_index, best_range, is_numerical )
    tree = decision_tree.left

    if gini(best_split[0]) == 0:
        print "sub left is pure"
        tree.left = Decision( best_split[0][0].label )

    if gini(best_split[1]) == 0:
        print "sub right is pure"
        tree.right = Decision( best_split[1][0].label )

    print "\n"
    print "Decision tree: ", decision_tree
    print "\n"

# current mci => yes
# otherwise:
    # phone usage < 75 => no
    # else => yes
    test_records = [
            Record( ['<75', '2', 'City', 'MCI'], 'Yes' ),
            Record( ['<75', '1', 'Town', 'AT&T'], 'No' ),
            Record( ['75..150', '1', 'Town', 'MCI'], 'Yes'),
            Record( ['75..150', '2', 'City', 'AT&T'], 'Yes')
            ]

    for r in test_records:
        print "Record", r
        print "Label:", r.label
        print "Predicted label:", decision_tree.vote( r )
    """

    """
    l,r = create_category_split( records, 1, (['MCI'], []) )
    #l,r = create_numerical_split( records, 0, 1 )
    print "Left"
    for a in l:
        print a
    print "Right"
    for a in r:
        print a

    print "Gini left: %s" % ( gini(l) )
    print "Gini right: %s" % ( gini(r) )
    print "Gini split: %s" % ( gini(records) - gini_split(l,r) )

    for split in generate_category_split(['A', 'B', 'C', 'D']):
        print split
    """


