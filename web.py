"""
    web.py
    Author: Benjamin Bouvier

    This file contains the script for the web front-end of the site.
"""
from tornado.ioloop import *
from tornado.web import *
from tornado.template import *
from tornado import autoreload

from MapReduce import launch_job
from Record import Record
import config

import csv
import json
import thread
import datetime

loader = template.Loader( config.MAIN_DIR )

def log(text):
    """Logs a message in the console."""
    print "%s: %s" % (str(datetime.datetime.now()), text)

class MainHandler(RequestHandler):
    """Handler for requests on the main page. Writes the template index.html
    in the socket."""
    def get(self, *args, **kwargs):
        log("Request on main handler")
        self.write( loader.load("index.html").generate() )

class DefaultValuesHandler(RequestHandler):
    """Handler for /default/*. Return record for default values, read from files. The files
    are read only once and then cached in memory."""
    def __init__(self, *args, **kwargs):
        super(DefaultValuesHandler, self).__init__(*args, **kwargs)
        self.cached_values = {}
        self.possible_values = dict({
                ('titanic', ('From Kaggle: can we predict if one will survive the sinking of the Titanic according to some personal informations?', 1000)),
                ('examples', ('From Random Forest example lecture note: will some people change phone plan according to their profile?', 1000)),
                ('digits', ('From Kaggle: can we predict which number is written if we only have the gray levels of the corresponding picture?', 100))
            })

    def __load(self, filename):
        """Loads a CSV file in memory."""
        values = {}
        training = []
        filecontent = loader.load('csv/' + filename + '.csv').generate().strip().split('\n')
        i = 0
        for r in filecontent:
            if i < 1000: # limit size of datasets to 1,000 records
                training.append( r )
                i += 1
            else:
                break

        values['training'] = '\n'.join(training)
        values['n_training'] = int(3. * len(training) / 4.)
        values['description'] = self.possible_values.get(filename, ('', 10))[0]
        values['n_trees'] = self.possible_values.get(filename, ('', 10))[1]
        self.cached_values[ filename ] = json.dumps(values)

    def get(self, *args, **kwargs):
        name = self.request.uri.split('/')[-1]

        if name not in self.possible_values:
            name = "examples"

        log("Getting example dataset with name: " + name)
        if name not in self.cached_values:
            self.__load(name)
        self.write( self.cached_values[ name ] )

class ForestHandler(RequestHandler):
    """Handler for post on /forest. Creates a job, parses the results and transmits them
    to the user.

    As the web server Tornado uses an event loop for processing events, we need to make the
    call to the cluster asynchroneous if we want to be able to visit other pages while the
    cluster is busy. For this purpose, we use the decorator @asynchronous and we launch the
    call to the cluster in another thread.
    """
    @asynchronous
    def post(self):

        # Never trust user input
        test_all = self.get_argument("test_all", 'off') == 'on'
        training = sorted(self.get_argument("training").strip().split('\n'))

        try:
            n_training = int( self.get_argument("n_training").strip() )
            assert (not test_all and n_training < len(training)) or n_training <= len(training)
            assert n_training > 0
        except: # if there is something wrong, just use 3/4 of the records for training purposes
            n_training = int(len(training)*3. / 4.)

        try:
            n_trees = int( self.get_argument("trees").strip() )
            assert n_trees >= 10
            if n_trees > 1000:
                n_trees = 1000
        except: # if there is something wrong, just grow 10 trees
            n_trees = 10

        log("job launched with %s records, %s trees" % (str(len(training)), str(n_trees)))
        def launch( training, n_training, n_trees, test_all, begin_date, req ):
            all_data = sorted(training)
            all_data_plain = '\n'.join(all_data).strip()

            results = launch_job(all_data_plain, n_training, n_trees, test_all, env=config.ENV)

            reader = csv.reader(all_data, delimiter=',')
            if test_all:
                records = [ Record( r[:-1], r[-1] ) for r in reader ]
            else:
                records = [ Record( r[:-1], r[-1] ) for r in list(reader)[n_training:] ]

            correct = 0
            max_labels = 0
            for k, v in results:
                votes = {}
                records[k].votes = v

                if len(v) > max_labels:
                    max_labels = len(v)

                if records[k].label == records[k].prediction():
                    records[k].correctly = True
                    correct += 1
                else:
                    records[k].correctly = False

                for lab in v:
                    votes[ lab ] = ( v[lab], (100.*v[lab])/n_trees )
                records[k].votes = votes

            total = len(records)

            stats = {}
            stats['correct'] = correct
            stats['correct_r'] = (100. * correct) / total
            stats['total'] = total

            stats['n_training'] = n_training
            stats['n_trees'] = n_trees
            stats['test_all'] = test_all
            stats['max_labels'] = max_labels
            now = datetime.datetime.now()
            stats['time'] = str(now - begin_date)

            req.write(loader.load("results.html").generate(stats=stats, results=records))
            log("MapReduce job finished (%s)" % stats['time'])
            req.finish()

        thread.start_new_thread( launch, (training, n_training, n_trees, test_all, datetime.datetime.now(), self) )

# Maps every URL (as a regular expression) to a handler
app = Application([
    (r'/', MainHandler),
    (r'/public/(.*)', StaticFileHandler, {"path": config.MAIN_DIR + "public/"}), # just serves static files contained in public
    (r'/default/(.*)', DefaultValuesHandler),
    (r'/forest', ForestHandler),
])

if __name__ == '__main__':
    app.listen(config.PORT)
    IOLoop.instance().start()
