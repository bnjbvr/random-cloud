class DecisionTree:
    def __init__(self, index, possible, is_numerical):
        self.criteria = index
        self.possible = possible
        self.numerical = is_numerical

        self.left = None
        self.right = None

    def vote(self, record):
        if self.numerical and float(record.features[self.criteria]) <= self.possible:
            next_voter = self.left
        elif not self.numerical and record.features[self.criteria] in self.possible:
            next_voter = self.left
        else:
            next_voter = self.right

        return next_voter.vote( record )

    def show(self, i):
        ret = "%d: feature(%s) in range %s? yes: %d / no: %d\n" % (i, str(self.criteria), str(self.possible), 2*i, 2*i+1)
        ret += self.left.show(2*i)
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


