"""
    unit_test_random_forest.py
    Author: Benjamin Bouvier

    Launches a Map / Reduce job for the random forest algorithm (locally,
    without using Hadoop or Elastic Map Reduce) and prints the results.
"""
from MapReduce import *
import json
import csv

if __name__ == '__main__':
    f = file('csv/digits.csv')
    strings = sorted(f.read().split('\n')[:1000])
    reader = csv.reader(strings, delimiter=',')
    records = [ Record(t[:-1], t[-1]) for t in reader ]
    ch = '\n'.join(strings)
    result = launch_job(ch, 1000, 20, True, env='hadoop')
    correct = 0
    for k, v in result:
        max, predicted = 0, None
        for l in v:
            if max < v[l]:
                max, predicted = v[l], l
        if predicted == records[k].label:
            correct += 1
        print "%s\tp:%s\tr:%s\t%s" % (k, predicted, records[k].label, json.dumps(v))
    print "TOTAL SCORE"
    print correct / 10.

