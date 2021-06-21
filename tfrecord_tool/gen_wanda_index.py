import os
import numpy as np

def get_index(txtpath):
    with open(txtpath) as wf:
        lines = wf.read().splitlines()
    print(txtpath, len(lines))
    camIDs, trackIDs = [], []
    for l in lines:
        name = os.path.basename(l)
        camID, trackID, time = name.split('_')
        camIDs.append(int(camID))
        trackIDs.append(int(trackID))
    mincamIDs = min(camIDs)
    mintrackIDs = min(trackIDs)
    camIDs = [c - mincamIDs for c in camIDs]
    #trackIDs = [t - mintrackIDs for t in trackIDs]
    labels = np.argsort(trackIDs).tolist()
    with open('indexes/index_' + os.path.basename(txtpath), 'w') as wf:
        for ln, l, c, t in zip(lines, labels, camIDs, trackIDs):
            wf.write('%s %s %s %s\n' % (ln, l, t, c))

if __name__ == '__main__':
    ds = ['dongguan_200903_200909', 'fuzhou_200903_200909', 'harbin_200903_200909', 'hebei_200903_200909', 'wuhan_200903_200909', 'zhengzhou_200903_200909']
    for d in ds:
        get_index('/dockerdata/fufuyu/wanda_reid_data/%s.txt' % d)
     


