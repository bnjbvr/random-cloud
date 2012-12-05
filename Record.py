class Record:
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.votes = {}
        self.picked = False
        self.correctly = True

    def __repr__(self):
        return ', '.join( self.features ) + ': ' + self.label

    def pick(self):
        self.picked = True

    def vote_for(self, value):
        self.votes[ value ] = self.votes.get( value, 0 ) + 1

    def prediction(self):
        p, max = None, 0
        for key in self.votes:
            if self.votes[key] > max:
                p, max = key, self.votes[key]
        return p
