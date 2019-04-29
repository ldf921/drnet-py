import os
import json

data_root = '../datasets/KTH'
vol = 5
with open(os.path.join(data_root, 'frames%d.txt' % vol)) as fi:
    frames = [line.strip() for line in fi]

with open(os.path.join(data_root, 'pose/vol%d/alphapose-results.json' % vol)) as fi:
    poses = json.load(fi)

vids = {}
for img_file in frames:
    vid = os.path.dirname(img_file)
    if vid not in vids:
        vids[vid] = 1
    else:
        vids[vid] += 1

for vid in vids:
    vids[vid] = {'frames' : vids[vid], 'poses' : set()}

for p in poses:
    img_file = p['image_full']
    vid = os.path.dirname(img_file).replace('../datasets/KTH/', '')
    f = vids[vid]
    f['poses'].add(os.path.basename(img_file))

for name, v in vids.items():
    print('{}  {:3d}  {:3d}'.format(name, v['frames'], len(v['poses'])))




