import csv
import math
from random import randint

#import cProfile # for profiling purposes only
#import guppy

from Record import *
from Dataset import *
from DecisionTree import *
from Split import *

from mrjob.job import MRJob

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

def grow_forest( n, records, test_records = None ):
    dataset = Dataset( records )
    record_number = dataset.size        # N

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

        if test_records == None or len(test_records) == 0:
            dts.append( tree )
        else:
            print "Voting", i
            for tr in test_records:
                tr.vote_for( tree.vote( tr ) )
            del tree

        del picked_records

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
    with open('titanic.csv', 'r') as csvfile:
    #with open('train2.csv', 'r') as csvfile:
    #with open('KDD.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if i < 1000:
                records.append( Record( row[:-1], row[-1] ) )
            else:
                test_records.append( Record( row[:-1], row[-1] ) )
            i += 1

    print "learning with %d records" % len(records)
    dts = grow_forest( 10, records, test_records )

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
        if total % 100 == 0:
            print "%s / %s correct (%s %%)" % (correct, total, correct / float(total) * 100.)
    print "%s / %s correct (%s %%)" % (correct, total, correct / float(total) * 100.)

    if len(test_records) > 0:
        print "On test records..."
        total, correct = 0, 0
        for r in test_records:
            total += 1
            if r.label == major_vote( dts, r ):
                correct += 1
            if total % 100 == 0:
                print "%s / %s correct (%s %%)" % (correct, total, correct / float(total) * 100.)
        print "%s / %s correct (%s %%)" % (correct, total, correct / float(total) * 100.)

class MRRandomForest(MRJob):

    def configure_options(self):
        super(MRRandomForest, self).configure_options()

        self.add_passthrough_option(
            '--trees', dest='tree_number', default=10, type='int',
            help='number of trees to create')

        self.add_passthrough_option(
            '--training', dest='training_records_number', default=100, type='int',
            help='number of training records to use')

        self.add_passthrough_option(
            '--test-all', dest='test_all', default=False, help='true if all records (training + testing) should be used for testing.', action = 'store_true')

    def __init__(self, **kwargs):
        super(MRRandomForest, self).__init__(**kwargs)

    def dispatcher(self, key, line):
        for i in range(self.options.tree_number): # number of trees
            yield i, line

    def tree_vote(self, tree_id, lines):
        lines = sorted(lines)
        reader = csv.reader( lines, delimiter=',' )

        testing_records = Dataset()
        training_records = Dataset()
        i = 0
        for row in reader:
            if i < self.options.training_records_number:
                training_records.append( Record( row[0:-1], row[-1] ) )
            else:
                testing_records.append( Record( row[0:-1], row[-1] ) )
            i += 1

        picked_records = []
        record_number = len(training_records)
        for j in xrange( record_number ):
            ind_picked = randint(0, record_number-1)
            picked_records.append( training_records[ ind_picked ] )
        picked_records = Dataset( picked_records )
        tree = train(picked_records)

        i = 0
        if self.options.test_all:
            for r in training_records:
                yield i, tree.vote(r)
                i += 1
        for r in testing_records:
            yield i, tree.vote(r)
            i += 1

    def repeater(self, record_id, tuple):
        yield record_id, tuple

    def aggregator(self, record_id, t):
        labels = {}
        for value in t:
            labels[ value ] = labels.get( value, 0 ) + 1
        yield record_id, labels

    def job_runner_kwargs(self):
        ret = super(MRRandomForest, self).job_runner_kwargs()
        ret['python_bin'] = '/usr/bin/pypy'
        return ret

    def hadoop_job_runner_kwargs(self):
        ret = super(MRRandomForest, self).hadoop_job_runner_kwargs()
        ret['python_archives'] = ['Record.py', 'Dataset.py', 'DecisionTree.py', 'Split.py']
        return ret

    def steps(self):
        return [self.mr(mapper=self.dispatcher, reducer=self.tree_vote), self.mr(mapper=self.repeater,
            reducer=self.aggregator)]

def launch_job(text, training_number, trees_number, test_all, env='local'):
    global features_type
    features_type = {}

    chosen_args = ['-v']

    if env != 'hadoop':
        text = text.split('\n')

    chosen_args.extend(['-r', env])

    chosen_args.append('--trees')
    if isinstance(trees_number, int):
        trees_number = str(trees_number)
    chosen_args.append(trees_number)

    if test_all:
        chosen_args.append('--test-all')

    chosen_args.append('--training')
    if isinstance(training_number, int):
        training_number = str(training_number)
    chosen_args.append(training_number)

    job = MRRandomForest(args=chosen_args)
    f = file('output', 'w+')
    job.sandbox(stdin=text)
    runner = job.make_runner()
    runner.run()

    result = []
    for line in runner.stream_output():
        key, value = job.parse_output_line(line)
        result.append( (key, value) )
    return result

if __name__ == '__main__':
    features_type = {}
    job = MRRandomForest()
    job.run()
