from Dataset import *

class Split:
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
        self.feature_index = index

        for r in records:
            side = self.right
            if self.is_numerical and float(r.features[ self.feature_index ]) <= self.feature_range:
                side = self.left
            elif not self.is_numerical and r.features[ self.feature_index ] in self.feature_range:
                side = self.left
            side.append( r )

        self.left.update()
        self.right.update()
        # compute gain
        self.left_gini = self.left.gini
        self.right_gini = self.right.gini
        l, r, n = self.left.size, self.right.size, float(records.size)
        self.gain = records.gini - (l/n)*self.left_gini - (r/n)*self.right_gini
