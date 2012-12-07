"""
Dataset.py
Author: Benjamin Bouvier

Represents a dataset of records. It is a list with augmented features.

Contains the value of Gini for this dataset,
the mode (= the most frequent label in the dataset),
the labels (= hashmap with count of the frequency of each label),
and two booleans label_monotone (indicating that all records have the same
label) and monotone (indicating that all records have the same feature).
"""

import math

class Dataset(list):
    def __init__(self, records = []):
        """Initializes the dataset from an existing list and computes all values."""
        super(Dataset, self).__init__(records)

        self.gini = None
        self.mode = None
        self.labels = {}
        self.label_monotone = None

        for r in records:
            self.labels[ r.label ] = self.labels.get( r.label, 0 ) + 1

        self.update()

    def append(self, record):
        """Overload of the list.append method. Updates the count of label."""
        super(Dataset, self).append( record )
        self.labels[ record.label ] = self.labels.get( record.label, 0 ) + 1

    def update(self):
        """Updates all values of gini, mode, labels, label_monotone and monotone."""
        self.size = len(self)

        # checks values are different
        self.monotone = self.all_same()

        # compute gini value and mode
        gini = 1.
        mode, max = None, 0
        for key in self.labels:
            gini -= math.pow( float(self.labels[key]) / self.size , 2 )
            if self.labels[key] > max:
                mode, max = key, self.labels[key]
        self.gini = gini
        self.mode = mode
        self.label_monotone = len( self.labels ) == 1

    def all_same(self):
        """Returns true if all records contain the exact same values for each feature."""
        if self.size == 0:
            return True
        else:
            comp = self[0]
            nb_feat = len(comp.features)
            for i in xrange(1, self.size):
                for j in xrange(nb_feat):
                    if comp.features[j] != self[i].features[j]:
                        return False
            return True


