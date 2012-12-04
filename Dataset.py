import math

class Dataset(list):
    def __init__(self, records = []):
        super(Dataset, self).__init__(records)

        self.gini = None
        self.mode = None
        self.labels = {}
        self.label_monotone = None

        for r in records:
            self.labels[ r.label ] = self.labels.get( r.label, 0 ) + 1

        self.update()

    def append(self, record):
        super(Dataset, self).append( record )
        self.labels[ record.label ] = self.labels.get( record.label, 0 ) + 1

    def update(self):
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


