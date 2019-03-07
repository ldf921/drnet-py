import os
import subprocess
import shlex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, help='data root')
parser.add_argument('--imageSize', type=int, help='data root')
args = parser.parse_args()
data_root = args.dataRoot
image_size = args.imageSize

classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

frame_rate = 25

for _, label in enumerate(classes):
    print('---')
    print(label)

    for vid in os.listdir(os.path.join(data_root,'raw', label)):
        print(vid)
        fname = vid.replace('_uncomp.avi', '')
        try:
            os.makedirs('%s/processed/%s/%s' % (data_root, label, fname))
        except OSError:
            pass
        subprocess.call(shlex.split('ffmpeg -i %s/raw/%s/%s -r %d -f image2 -s %dx%d  %s/processed/%s/%s/image-%%03d_%dx%d.png' % (data_root, label, vid, frame_rate, image_size, image_size, data_root, label, fname, image_size, image_size)))

