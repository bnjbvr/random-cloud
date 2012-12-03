import csv
import math
from random import randint

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

class Record:
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.picked = False

    def __repr__(self):
        return ', '.join( self.features ) + ': ' + self.label

    def pick(self):
        self.picked = True

class Dataset:
    def __init__(self, records = []):
        self.records = records
        self.size = len(records)
        self.gini_value = None

    def gini(self):
        if self.gini_value is None:
            labels = {}
            for r in self.records:
                labels[ r.label ] = labels.get( r.label, 0. ) + 1.
            gini = 1.
            for key in labels:
                gini -= math.pow( labels[key] / self.size , 2 )
            self.gini_value = gini

        return self.gini_value

    def __iter__(self):
        for r in self.records:
            yield r

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return self.size

def feature_is_numerical(records, index):
    """Returns true if all values of the features at the given index are numerical in the
    given set of records, false otherwise."""
    for r in records:
        try:
            a = float(r.features[index])
        except:
            return False
    return True

class Split:
    def __init__(self, is_numerical):
        self.is_numerical = is_numerical

        self.left = []
        self.right = []

        self.feature_index = None
        if is_numerical:
            self.feature_range = None
        else:
            self.feature_range = {}

        self.gain = -1

    def add_category_range( self, value ):
        self.feature_range[ value ] = True

    def set_numerical_range( self, value ):
        self.feature_range = value

    def set_index(self, index):
        self.feature_index = index

    def fill(self, records):
        for r in records:
            side = self.right
            if self.is_numerical and float(r.features[ self.feature_index ]) <= self.feature_range:
                side = self.left
            if not self.is_numerical and r.features[ self.feature_index ] in self.feature_range:
                side = self.left
            side.append( r )

        self.left = Dataset( self.left )
        self.right = Dataset( self.right )

        # compute gain
        self.left_gini = self.left.gini()
        self.right_gini = self.right.gini()
        l, r, n = self.left.size, self.right.size, float(records.size)
        self.gain = records.gini() - (l/n)*self.left_gini - (r/n)*self.right_gini

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

"""
def create_category_split(records, index, split):
    left, right = [], []
    for r in records:
        if r.features[index] in split[0]:
            left.append(r)
        else:
            right.append(r)
    return index, split[0], left, right
"""

def generate_category_splits( records, index ):
    possible = {}
    for r in records:
        possible[ r.features[index] ] = True
    possible = possible.keys()
    splits = []
    for choice in generate_category_choice( possible ):
        #splits.append( create_category_split( records, index, choice ) )
        choice.set_index( index )
        choice.fill( records )
        splits.append( choice )
    return splits

"""
def create_numerical_split(records, index, split_value):
    left, right = [], []
    for r in records:
        if int(r.features[index]) <= split_value:
            left.append(r)
        else:
            right.append(r)
    return index, split_value, left, right
"""

def generate_numerical_splits( records, index ):
    possible = {}
    for r in records:
        possible[ r.features[index] ] = True
    possible = possible.keys()
    splits = []

    for i in xrange(0, len(possible)-1):
        s = Split(is_numerical=True)
        s.set_index( index )
        s.fill( records )
        splits.append( s )
        #splits.append( create_numerical_split( records, index, int(possible[i]) ) )
    #splits.append( (index, possible[0], records, []) )

    return splits

def generate_splits( records, index ):
    splits = []
    is_numerical = feature_is_numerical( records, index )
    #print "FEATURE", index, "NUMERICAL?", is_numerical
    if is_numerical:
        splits = generate_numerical_splits( records, index )
    else:
        splits = generate_category_splits( records, index )

    #print "Len splits",len(splits)
    if len(splits) == 0:
# happens when all records have the same value in the feature at given index
        #splits = [(index, records[0].features[index], records, [])]
        split = Split( is_numerical=is_numerical )
        split.set_index( index )
        if is_numerical:
            split.feature_range = records[0].features[index]
        else:
            split.feature_range = [records[0].features[index]]
        split.left = records
        split.right = []
        splits = [ split ]

    return splits, is_numerical

class DecisionTree:
    def __init__(self, index, possible, is_numerical):
        self.criteria = index
        self.possible = possible
        self.numerical = is_numerical

        self.left = None
        self.right = None

    def vote(self, record):
        next_voter = self.right
        if self.numerical and float(record.features[self.criteria]) <= self.possible:
            next_voter = self.left
        elif not self.numerical and record.features[self.criteria] in self.possible:
            next_voter = self.left

        return next_voter.vote( record )

    def __repr__(self):
        ret = ""
        if self.left is not None:
            ret += self.left.__repr__() + ', '
        ret += '[' + str(self.criteria) + ' in ' + str(self.possible) + ']'
        if self.right is not None:
            ret += ', ' + self.right.__repr__()
        return ret

class Decision:
    def __init__(self, prediction):
        self.prediction = prediction

    def __repr__(self):
        return "(Decision: " + str(self.prediction) + ')'

    def vote(self, record):
        return self.prediction

def train(records):
    m = len( records[0].features )
    sqm = int(math.sqrt(m))
    return train_r( records, range(m), sqm )

def train_r(records, attributes, sqm):
    chosen_attributes = []

    if len(attributes) >= sqm:
        while chosen_attributes == [] or len(list(set(chosen_attributes))) != len(chosen_attributes):
            chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]
    else:
        chosen_attributes = attributes

    if len(attributes) == 0 or len(records) == 0:
        #print "LIMIT CASE"
        vote = {}
        for r in records:
            vote[ r.label ] = vote.get( r.label, 0 ) + 1
        max = 0
        best = None
        for p in vote:
            if best == None or vote[p] > max:
                max, best = vote[p], p
        return Decision( best )

    #print "chosen attributes:"
    #for a in chosen_attributes:
    #    print a,
    #print '\n'

    best_gain = None
    best_split = None

    best_index = None
    best_range = None
    is_numerical = None

    for criteria in chosen_attributes:
        splits, isnum = generate_splits( records, criteria )
        for s in splits:
        #for (meta_ind, meta_range, left, right) in splits:
            gain = s.gain
            #print "SPLIT:\n%s\n%s" % ( left, right )
            #print "Gain:", gain
            #print "\n"
            if best_gain is None or best_gain < gain:
                best_split = s
                best_gain = gain

    #print "Best split:", best_split[0], '\n', best_split[1]
    #print "obtained for attribute @ index", best_index
    #print "with range ", best_range

    decision_tree = DecisionTree( s.feature_index, s.feature_range, s.is_numerical )
    if s.right_gini == 0:
    #if gini(best_split[1]) == 0:
        prediction = s.right[0].label
        #prediction = best_split[1][0].label
        #print "Right subtree is pure"
        #print "Prediction would be:", prediction
        decision_tree.right = Decision( prediction )
    else:
        new_attributes = attributes[:]
        new_attributes.remove( s.feature_index )
        decision_tree.right = train_r( s.right, new_attributes, sqm )
        #decision_tree.right = train_r( best_split[1], new_attributes, sqm )

    if s.left_gini == 0:
    #if gini(best_split[0]) == 0:
        prediction = s.left[0].label
        #print "Left subtree is pure"
        #print "Prediction would be:", prediction
        decision_tree.left = Decision( prediction )
    else:
        new_attributes = attributes[:]
        new_attributes.remove( s.feature_index )
        decision_tree.left= train_r( s.left, new_attributes, sqm )

    return decision_tree

def grow_forest( n, records ):
    dataset = Dataset( records )
    record_number = dataset.size        # N
    features_number = len(records[0].features)   # M

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
        dts.append( train(picked_records) )

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
    return k

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

def main():
    records = []
    #with open('examples.csv', 'r') as csvfile:
    with open('train2.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            records.append( Record( row[:-1], row[-1] ) )

    """
    record_number = len(records)        # N
    features_number = len(records[0].features)   # M

    picked_records = []
    for i in xrange( record_number ):
        ind_picked = randint(0, record_number-1)
        records[ ind_picked ].pick()
        picked_records.append( records[ ind_picked ] )

    dt = train(picked_records)
    for p in picked_records:
        prediction = dt.vote(p)
        if prediction != p.label:
            print "FATAL ERROR"
            print "p label:", p.label
            print "prediction:", prediction

    test_records = []
    correct = 0
    for r in records:
        if not r.picked:
            test_records.append( r )
            prediction = dt.vote(r)
            print "Test: real label =", r.label, ", predicted =", prediction
            if prediction == r.label:
                correct += 1

    print "%s correct out of %s" % (correct, len(test_records))
    """

    dts = grow_forest( 500, records )
    correct, total = 0, 0
    for r in records:
        #if not r.picked:
            total += 1
            if r.label == major_vote( dts, r ):
                correct += 1
    print "%s / %s correct" % (correct, total)
    print "%s" % ( correct / float(total) * 100. )


if __name__ == '__main__':
    main()
    #MRWordCounter.run()
