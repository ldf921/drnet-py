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

frame_rate = 10

for vid in os.listdir(os.path.join(data_root, 'raw')):
    print(vid)
    fname = vid.replace('.avi', '').replace('v_', '')
    try:
        os.makedirs('%s/processed/%s' % (data_root, fname))
    except OSError:
        pass
    subprocess.check_call(shlex.split('ffmpeg -i %s/raw/%s -r %d -f image2 -s %dx%d  %s/processed/%s/image-%%03d_%dx%d.png' % (data_root, vid, frame_rate, image_size, image_size, data_root, fname, image_size, image_size)))

