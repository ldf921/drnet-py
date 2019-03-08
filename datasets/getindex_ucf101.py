import os
import subprocess
import shlex
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, help='data root')
parser.add_argument('--imageSize', type=int, help='data root')
args = parser.parse_args()
data_root = args.dataRoot
image_size = args.imageSize

with open(os.path.join(data_root, 'classInd.txt'), 'r') as fi:
    tokens = [ln.split() for ln in fi.readlines()]
    class_index = {name.lower() : int(y) - 1 for y, name in tokens}

def generate_video_list(prefix, split_file):
    with open(split_file, 'r') as fi:
        if 'train' in split_file:
            tokens = [ln.strip().split() for ln in fi.readlines()]
            tokens = [(x.split('/')[1][2:-4], int(y) - 1) for x, y in tokens]
        else:
            tokens = [ln.strip() for ln in fi.readlines()]
            tokens = [x.split('/')[1][2:-4] for x in tokens]
            tokens = [ (x, class_index[x.split('_')[0].lower()]) for x in tokens ]

        return [ {'files' : os.listdir(os.path.join(prefix, vid)), 'vid' : vid, 'label' : y} for vid, y in tokens]

indexfolder = os.path.join(data_root, 'frames_%dx%d' % (image_size, image_size))
os.makedirs(indexfolder, exist_ok=True)

for i in range(1, 4):
    for subset in ('train', 'test'):
        flist = generate_video_list(
                os.path.join(data_root, 'processed'),
                os.path.join(data_root, '%slist0%d.txt' % (subset, i)))
        with open(os.path.join(indexfolder, '%s%d.pkl' % (subset, i)), 'wb') as fo:
            pickle.dump(flist, fo)

