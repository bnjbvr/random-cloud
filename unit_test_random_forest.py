from randomforest import *
import json

if __name__ == '__main__':
    f = file('csv/titanic.csv')
    ch = '\n'.join(f.read().split('\n')[:-1])
    result = launch_job(ch, 500, 20, True)
    for k, v in result:
        print "%s\t%s" % (k, json.dumps(v))

