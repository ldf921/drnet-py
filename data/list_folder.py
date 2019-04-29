import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output', type=str)
args = parser.parse_args()

data_root = '/home/ubuntu/disk/datasets/KTH'
output_file = args.output
vol = 0
fo = open(output_file + '%d.txt' % vol, 'w')
VOL_SIZE = 60
i = 0
for root, dirs, files in os.walk(data_root):
    sroot = root.replace(data_root + "/", "")
    rootdirs = sroot.split("/")
    if len(rootdirs) == 3:
        print(root)
        i += 1
        for f in files:
            fo.write(sroot + '/' + f + "\n")
        if i % VOL_SIZE == 0:
            fo.close()
            vol += 1
            fo = open(output_file + '%d.txt' % vol, 'w')
fo.close()

