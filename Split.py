"""
    Split.py
    Author: Benjamin Bouvier
"""
from Dataset import *

class Split:
    """This class is the model of a split.
    While it is very similar to a DecisionTree, the main difference is that the
    split contains the real associated left and right datasets.

    Moreover, the Split method contains also a method allowing easily to compute
    the gain of the split.

    @see DecisionTree.py
    """
    def __init__(self, is_numerical):
        self.is_numerical = is_numerical

        self.left = Dataset()
        self.right = Dataset()

        self.feature_index = None
        if is_numerical:
            self.feature_range = None
        else:
            self.feature_range = {}

        self.gain = -1

    def add_category_range( self, value ):
        self.feature_range[ value ] = True

    def set_numerical_range( self, value ):
        self.feature_range = float(value)

    def place(self, records, index):
        """Puts the records in the good side, with respect to the feature present at the given index.

        Also updates value of gini and gain
        """
        self.feature_index = index

        for r in records:
            if self.is_numerical and float(r.features[ self.feature_index ]) <= self.feature_range:
                side = self.left
            elif not self.is_numerical and r.features[ self.feature_index ] in self.feature_range:
                side = self.left
            else:
                side = self.right
            side.append( r )

        self.left.update()
        self.right.update()
        # compute gain
        self.left_gini = self.left.gini
        self.right_gini = self.right.gini
        l, r, n = self.left.size, self.right.size, float(records.size)
        self.gain = records.gini - (l/n)*self.left_gini - (r/n)*self.right_gini
