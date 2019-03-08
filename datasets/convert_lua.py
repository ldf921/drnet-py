import os
import argparse
import torchfile
import pickle


def convert_objects(obj):
    if hasattr(obj, 'keys'):
        return { k.decode() : convert_objects(obj[k]) for k in obj.keys() }
    elif isinstance(obj, list):
        return [ convert_objects(v) for v in obj ]
    elif isinstance(obj, bytes):
        return obj.decode()
    else:
        return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type=str, help='data root')
    args = parser.parse_args()
    data_root = args.dataRoot

    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

    for _, label in enumerate(classes):
        print('---')
        print(label)

        prefix = os.path.join(data_root,'processed', label)
        for fname in os.listdir(prefix):
            if fname.endswith('.t7'):
                obj = torchfile.load(os.path.join(prefix, fname))
                with open(os.path.join(prefix, fname.replace('.t7', '.pkl')), 'wb') as fo:
                    pickle.dump(convert_objects(obj), fo)
