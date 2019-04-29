import argparse
import os
import json
import pickle as pkl

parser = argparse.ArgumentParser()
args = parser.parse_args()

data_root = '../datasets/KTH'

vids = {}
for vol in range(0, 10):
    print('Voume ', vol)
    with open(os.path.join(data_root, 'frames%d.txt' % vol)) as fi:
        frames = [line.strip() for line in fi]

    with open(os.path.join(data_root, 'pose/vol%d/alphapose-results.json' % vol)) as fi:
        poses = json.load(fi)

    for img_file in frames:
        vid = os.path.basename(os.path.dirname(img_file))
        if vid not in vids:
            vids[vid] = {'frames'  : 1, 'poses' : dict()}
        else:
            vids[vid]['frames'] += 1


    for p in poses:
        img_file = p['image_full']
        vid = os.path.basename(os.path.dirname(img_file))
        f = vids[vid]
        frame_id = os.path.basename(img_file)
        p.pop('image_full')
        p.pop('image_id')
        if frame_id not in f['poses']:
            f['poses'][frame_id] = p
        elif f['poses'][frame_id]['score'] < p['score']:
            f['poses'][frame_id] = p

for name, v in vids.items():
    print('{}  {:3d}  {:3d}'.format(name, v['frames'], len(v['poses'])))


for class_name in os.listdir(os.path.join(data_root, 'processed')):
    class_root = os.path.join(data_root, 'processed', class_name)
    print(class_root)
    for subset in ('train', 'test'):
        with open(os.path.join(class_root, '%s_meta128x128.pkl') % subset, 'rb') as fi:
            videos = pkl.load(fi)
        for video in videos:
            vposes = vids[video['vid']]['poses']
            poses = []
            for frameslist in video['files']:
                poses.append([vposes.get(frame, None) for frame in frameslist])
            video['poses'] = poses
        with open(os.path.join(class_root, '%s_pose128x128.pkl') % subset, 'wb') as fo:
            pkl.dump(videos, fo)



