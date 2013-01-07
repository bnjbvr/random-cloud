"""
    MapReduce.py
    Author: Benjamin Bouvier

    Defines the Map / Reduce job itself.
"""
from RandomForest import *
from mrjob.job import MRJob

class MRRandomForest(MRJob):
    """The Random Forest Map / Reduce class defines two map reduce jobs.
    It takes as an input the records as text lines and returns all votes for each record, as
    a hashtable.

    First MapReduce:
    - map: for each tree, send the line.
    - reduce: grow a tree and for each record, return a vote.

    Second MapReduce:
    - map: just repeat the vote.
    - reduce: for each record, aggregate votes.
    """

    def configure_options(self):
        """Command line options."""
        super(MRRandomForest, self).configure_options()

        self.add_passthrough_option(
            '--trees', dest='tree_number', default=10, type='int',
            help='number of trees to create')

        self.add_passthrough_option(
            '--training', dest='training_records_number', default=100, type='int',
            help='number of training records to use')

        self.add_passthrough_option(
            '--test-all', dest='test_all', default=False,
            help='true if all records (training + testing) should be used for testing.'
            , action = 'store_true')

    def __init__(self, **kwargs):
        super(MRRandomForest, self).__init__(**kwargs)

    def dispatcher(self, key, line):
        """Map of first job: transmit lines, for each tree"""
        for i in range(self.options.tree_number): # number of trees
            yield i, line

    def tree_vote(self, tree_id, lines):
        """Reduce of first job.
        This function is called for a fixed tree_id. The corresponding function receives all
        records and can use all of them. The tree is grown and the votes are emitted."""
	try:
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

		tree = grow_forest( 1, training_records )[0]

		i = 0
		if self.options.test_all:
		    for r in training_records:
			yield i, tree.vote(r)
			i += 1
		for r in testing_records:
		    yield i, tree.vote(r)
		    i += 1
	except:
		print "ERROR WHEN COMPUTING TREE: \n", "\n".join(lines)

    def repeater(self, record_id, tuple):
        """Map of the second job: just repeat"""
        yield record_id, tuple

    def aggregator(self, record_id, t):
        """Reduce of the second job: aggregates the votes"""
        labels = {}
        for value in t:
            labels[ value ] = labels.get( value, 0 ) + 1
        yield record_id, labels

    def job_runner_kwargs(self):
        """Options to pass to mrjob.
        Just changes the interpreter to use pypy."""
        ret = super(MRRandomForest, self).job_runner_kwargs()
        ret['python_bin'] = '/usr/bin/pypy'
        return ret

    def hadoop_job_runner_kwargs(self):
        """Options to pass to mrjob when using Hadoop.

        - python_archives allows to send all the python scripts to Hadoop, so that they
        can be imported.
        - jobconf allows to specify the number of map tasks and reduce tasks for the job.
        As there are 4 workers (slaves), these values are multiple of 4 here.
        """
        ret = super(MRRandomForest, self).hadoop_job_runner_kwargs()
        ret['python_archives'] = ['Record.py', 'Dataset.py', 'DecisionTree.py', 'Split.py', 'RandomForest.py']
        ret['jobconf'] = {'mapred.map.tasks': '4', 'mapred.reduce.tasks': '8'}
        return ret

    def steps(self):
        """Defines the steps of the Map Reduce job."""
        return [self.mr(mapper=self.dispatcher, reducer=self.tree_vote), self.mr(mapper=self.repeater,
            reducer=self.aggregator)]

def launch_job(text, training_number, trees_number, test_all, env='local'):
    """Using the text containing all records separated by lines in the CSV format,
    launches a Map / Reduce job, growing 'trees_number' trees, using 'training_number' records
    for each tree. The used environment is env and can be 'local', 'inline', 'emr' or 'hadoop';
    these are the possible mrjob command line options.

    Applies the decision process for testing records (and training records if test_all is True)."""
    global features_type
    features_type = {} # reinitializes the map for features type.

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
