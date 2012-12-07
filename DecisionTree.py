"""
DecisionTree.py
Author: Benjamin Bouvier

This file contains the definitions of DecisionTree and Decision.
"""
class DecisionTree:
    """
    Represents a DecisionTree internal node.
    The associated value is a triplet (criteria, possible values, is numerical) which
    allows to redirect the input record to a sub-tree:
    if the record's feature with index 'criteria' is in the range 'possible values',
    then we put the record in the left sub-tree, otherwise we put it in the right sub-tree.
    The attribute is_numerical allows to distinct numerical and nominal features.
    """
    def __init__(self, index, possible, is_numerical):
        self.criteria = index
        self.possible = possible
        self.numerical = is_numerical

        self.left = None
        self.right = None

    def vote(self, record):
        """Votes for a record, redirecting to the left or the right subtree."""
        if self.numerical and float(record.features[self.criteria]) <= self.possible:
            next_voter = self.left
        elif not self.numerical and record.features[self.criteria] in self.possible:
            next_voter = self.left
        else:
            next_voter = self.right

        return next_voter.vote( record )

    def show(self, i):
        """Readable way to print a tree."""
        ret = "%d: feature(%s) in range %s? yes: %d / no: %d\n" % (i, str(self.criteria), str(self.possible), 2*i, 2*i+1)
        ret += self.left.show(2*i)
        ret += self.right.show(2*i+1)
        return ret

    def __repr__(self):
        return self.show(1)

class Decision:
    """A decision is a leaf of a DecisionTree. It just returns the name of the predicted label."""
    def __init__(self, prediction):
        if prediction is None:
            print "PREDICTION is none" # this should never happen
        self.prediction = prediction

    def show(self, i):
        return "%d: label %s\n" % (i, str(self.prediction))

    def __repr__(self):
        return self.show(1)

    def vote(self, record):
        return self.prediction

