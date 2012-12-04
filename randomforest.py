import csv
import cProfile
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

        self.gini = None
        self.mode = None
        self.labels = {}
        self.label_monotone = None

        # checks values are different
        self.monotone = self.all_same()

        # compute gini value
        labels = {}
        for r in self.records:
            labels[ r.label ] = labels.get( r.label, 0. ) + 1.
        gini = 1.
        for key in labels:
            gini -= math.pow( labels[key] / self.size , 2 )
        self.gini = gini

        # computing mode
        mode, max = None, 0.
        for key in labels:
            if labels[key] > max:
                mode, max = key, labels[key]
        self.mode = mode

        self.label_monotone = len( labels ) == 1

    def all_same(self):
        if self.size == 0:
            return True
        else:
            comp = self.records[0]
            for i in range(1, len(self.records)):
                for j in range(len(comp.features)):
                    if comp.features[j] != self.records[i].features[j]:
                        return False
            return True

    def __iter__(self):
        for r in self.records:
            yield r

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return self.size


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
                break
    return features_type[index]

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

    def place(self, records):
        for r in records:
            side = self.right
            if self.is_numerical and float(r.features[ self.feature_index ]) <= self.feature_range:
                side = self.left
            elif not self.is_numerical and r.features[ self.feature_index ] in self.feature_range:
                side = self.left
            side.append( r )

        self.fill(records, self.left, self.right)

    def fill(self, records, left, right ):
        self.left = Dataset( left )
        self.right = Dataset( right )

        # compute gain
        self.left_gini = self.left.gini
        self.right_gini = self.right.gini
        l, r, n = self.left.size, self.right.size, float(records.size)
        self.gain = records.gini - (l/n)*self.left_gini - (r/n)*self.right_gini

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
        choice.set_index( index )
        choice.place( records )
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
        s.set_index( index )
        s.set_numerical_range( i )
        s.place( records )
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

    #print "Len splits",len(splits)
    """
    if len(splits) == 0:
# happens when all records have the same value in the feature at given index
        #splits = [(index, records[0].features[index], records, [])]
        split = Split( is_numerical=is_numerical )
        split.set_index( index )
        if is_numerical:
            split.feature_range = records[0].features[index]
        else:
            split.feature_range = [records[0].features[index]]
        split.fill( records, records, [] )
        splits = [ split ]
    """

    return splits

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

    def show(self, i):
        ret = "%d: feature(%s) in range %s? yes: %d / no: %d\n" % (i, str(self.criteria), str(self.possible), 2*i, 2*i+1)
        #if self.left is not None:
        ret += self.left.show(2*i)
        #if self.right is not None:
        ret += self.right.show(2*i+1)
        return ret

    def __repr__(self):
        return self.show(1)

class Decision:
    def __init__(self, prediction):
        if prediction is None:
            print "PREDICTION is none"
        self.prediction = prediction

    def show(self, i):
        return "%d: label %s\n" % (i, str(self.prediction))

    def __repr__(self):
        return self.show(1)

    def vote(self, record):
        return self.prediction

def train(records):
    m = len( records[0].features )
    sqm = int(math.sqrt(m))
    return train_r( records, range(m), sqm, 1e30)

def train_r(records, attributes, sqm, depth):

    if records.label_monotone or records.monotone or depth == 0:
        return Decision(records.mode)

    depth -= 1
    chosen_attributes = []
    attributes_with_no_split = 0

    #print "Attributes: ",attributes
    while len(chosen_attributes) == attributes_with_no_split:
        chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]
        #if len(attributes) >= sqm:
        while len(list(set(chosen_attributes))) != len(chosen_attributes):
            chosen_attributes = [attributes[randint(0, len(attributes)-1)] for i in xrange(sqm)]
        #else:
        #    chosen_attributes = attributes

        if len(attributes) == 0:
            print "LIMIT CASE ATTR SIZE"
            return Decision( records.mode )

        #print "chosen attributes:"
        #for a in chosen_attributes:
        #    print a,

        best_gain = -1
        best_split = None

        best_index = None
        best_range = None
        is_numerical = None

        attributes_with_no_split = 0
        for criteria in chosen_attributes:
            splits = generate_splits( records, criteria )
            if len(splits) == 0:
                #print "LEN SPLITS = 0"
                attributes_with_no_split += 1

            for s in splits:
            #for (meta_ind, meta_range, left, right) in splits:
                gain = s.gain
                #print "SPLIT:\n%s\n%s" % ( left, right )
                #print "Gain:", gain
                #print "\n"
                if best_gain < gain:
                    best_split = s
                    best_gain = gain

        #print "attributes with no split = ", attributes_with_no_split, " / len(chosen) = ", len(chosen_attributes)
        #print "records:"
        #for r in records:
        #    print r

    #print "best split obtained for attribute @ index", best_split.feature_index
    #print "with range ", best_split.feature_range

    s = best_split
    decision_tree = DecisionTree( s.feature_index, s.feature_range, s.is_numerical )

    if s.left.size == 0 or s.right.size == 0:
        #print "LIMIT CASE SIZE"
        return Decision( records.mode )

    if s.right.label_monotone or s.right.monotone:
        decision_tree.right = Decision( s.right.mode )
    #if gini(best_split[1]) == 0:
        #prediction = best_split[1][0].label
        #print "Right subtree is pure"
        #print "Prediction would be:", prediction
    else:
        new_attributes = attributes[:]
        #new_attributes.remove( s.feature_index )
        decision_tree.right = train_r( s.right, new_attributes, sqm, depth )
        #decision_tree.right = train_r( best_split[1], new_attributes, sqm )

    if s.left.monotone or s.left.label_monotone:
        decision_tree.left = Decision( s.left.mode )
    #if gini(best_split[0]) == 0:
        #print "Left subtree is pure"
        #print "Prediction would be:", prediction
    else:
        new_attributes = attributes[:]
        #new_attributes.remove( s.feature_index )
        decision_tree.left= train_r( s.left, new_attributes, sqm, depth )

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

    #print dts
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

def main():
    records = []
    #with open('examples.csv', 'r') as csvfile:
    #with open('titanic.csv', 'r') as csvfile:
    with open('train2.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            records.append( Record( row[:-1], row[-1] ) )

    dts = grow_forest( 5, records )
    correct, total = 0, 0

    """
    for dt in dts:
        print 'dt:', dt
    """

    for r in records:
        #if not r.picked:
            total += 1
            if r.label == major_vote( dts, r ):
                correct += 1
    print "%s / %s correct" % (correct, total)
    print "%s" % ( correct / float(total) * 100. )


if __name__ == '__main__':
    #main()
    cProfile.run('main()')
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


