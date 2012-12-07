"""
    unit_test_seq_random_forest.py
    Author: Benjamin Bouvier

    Runs the algorithm works in a sequential mode (not distributed) and
    computes performances.
"""
from RandomForest import *
import random

def print_score( correct, total ):
    print "%s / %s correct (%s %%)" % (correct, total, correct / float(total) * 100.)

def test_collection( records, dts ):
    correct, total = 0, 0
    for r in records:
        if total > 0 and total % 100 == 0:
            print_score(correct, total)
        total += 1
        if r.label == major_vote( dts, r ):
            correct += 1
    print_score(correct, total)

def main():
    files = ['examples', 'titanic', 'digits', 'KDD']

    N_TRAINING = 1000
    N_TREES = 100
    CHOSEN_FILE = files[2]
    PRINT_TREES = False
    TEST = True

    records = []
    test_records = []
    i = 0
    with open('csv/' + CHOSEN_FILE + '.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if i < N_TRAINING:
                records.append( Record( row[:-1], row[-1] ) )
            else:
                test_records.append( Record( row[:-1], row[-1] ) )
            i += 1

    print "learning with %d records" % len(records)
    random.shuffle( records )
    dts = grow_forest( N_TREES, records )

    if PRINT_TREES:
        for dt in dts:
            print 'dt:', dt

    print "On records used for learning..."
    test_collection( records, dts )

    if len(test_records) > 0 and TEST:
        print "On test records..."
        test_collection( test_records, dts )

if __name__ == '__main__':
    main()
