"""
Record.py
Author: Benjamin Bouvier
"""
class Record:
    """Represents a record in the database.
    It contains
    - the list of the features
    - the label
    - a hashmap containing, for each possible predicted label, the number of
    votes for this label
    - correctly is a boolean indicating whether the record has been correctly
    predicted or not.
    """
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.votes = {}
        self.correctly = True

    def __repr__(self):
        return ', '.join( self.features ) + ': ' + self.label

    def vote_for(self, value):
        """Gives one more vote to the label 'value'"""
        self.votes[ value ] = self.votes.get( value, 0 ) + 1

    def prediction(self):
        """Returns the label with the biggest amount of votes."""
        p, max = None, 0
        for key in self.votes:
            if self.votes[key] > max:
                p, max = key, self.votes[key]
        return p
