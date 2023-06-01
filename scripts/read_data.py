from dblog import DbLog
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('file')
args = parser.parse_args()
data = DbLog(args.file)
results = np.array(data.read('eval/success_rate_out_dist')).astype(float)
for it,sr in results:
    print("iteration: ",int(it)," mean success rate on held out tasks: ",sr)


