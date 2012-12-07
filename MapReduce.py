from RandomForest import *
from mrjob.job import MRJob

class MRRandomForest(MRJob):
    def configure_options(self):
        super(MRRandomForest, self).configure_options()

        self.add_passthrough_option(
            '--trees', dest='tree_number', default=10, type='int',
            help='number of trees to create')

        self.add_passthrough_option(
            '--training', dest='training_records_number', default=100, type='int',
            help='number of training records to use')

        self.add_passthrough_option(
            '--test-all', dest='test_all', default=False, help='true if all records (training + testing) should be used for testing.', action = 'store_true')

    def __init__(self, **kwargs):
        super(MRRandomForest, self).__init__(**kwargs)

    def dispatcher(self, key, line):
        for i in range(self.options.tree_number): # number of trees
            yield i, line

    def tree_vote(self, tree_id, lines):
        lines = sorted(lines)
        reader = csv.reader( lines, delimiter=',' )

        testing_records = []
        training_records = []
        i = 0
        for row in reader:
            if i < self.options.training_records_number:
                training_records.append( Record( row[0:-1], row[-1] ) )
            else:
                testing_records.append( Record( row[0:-1], row[-1] ) )
            i += 1

        tree = grow_forest( 1, training_records, None )[0]

        i = 0
        if self.options.test_all:
            for r in training_records:
                yield i, tree.vote(r)
                i += 1
        for r in testing_records:
            yield i, tree.vote(r)
            i += 1

    def repeater(self, record_id, tuple):
        yield record_id, tuple

    def aggregator(self, record_id, t):
        labels = {}
        for value in t:
            labels[ value ] = labels.get( value, 0 ) + 1
        yield record_id, labels

    def job_runner_kwargs(self):
        ret = super(MRRandomForest, self).job_runner_kwargs()
        ret['python_bin'] = '/usr/bin/pypy'
        return ret

    def hadoop_job_runner_kwargs(self):
        ret = super(MRRandomForest, self).hadoop_job_runner_kwargs()
        ret['python_archives'] = ['Record.py', 'Dataset.py', 'DecisionTree.py', 'Split.py', 'RandomForest.py']
        ret['jobconf'] = {'mapred.map.tasks': '8', 'mapred.reduce.tasks': '8'}
        return ret

    def steps(self):
        return [self.mr(mapper=self.dispatcher, reducer=self.tree_vote), self.mr(mapper=self.repeater,
            reducer=self.aggregator)]

def launch_job(text, training_number, trees_number, test_all, env='local'):
    global features_type
    features_type = {}

    chosen_args = ['-v']

    if env != 'hadoop':
        text = text.split('\n')

    chosen_args.extend(['-r', env])

    chosen_args.append('--trees')
    if isinstance(trees_number, int):
        trees_number = str(trees_number)
    chosen_args.append(trees_number)

    if test_all:
        chosen_args.append('--test-all')

    chosen_args.append('--training')
    if isinstance(training_number, int):
        training_number = str(training_number)
    chosen_args.append(training_number)

    job = MRRandomForest(args=chosen_args)
    f = file('output', 'w+')
    job.sandbox(stdin=text)
    runner = job.make_runner()
    runner.run()

    result = []
    for line in runner.stream_output():
        key, value = job.parse_output_line(line)
        result.append( (key, value) )
    return result

if __name__ == '__main__':
    features_type = {}
    job = MRRandomForest()
    job.run()
