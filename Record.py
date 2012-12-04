class Record:
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.picked = False

    def __repr__(self):
        return ', '.join( self.features ) + ': ' + self.label

    def pick(self):
        self.picked = True
