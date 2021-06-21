import os
import glob

def select_file(path):
    with open(path) as rf:
        lines = rf.read().splitlines()
    new_lines = []
    classes = set()
    for l in lines:
        time = l.split('/')[1]
        if time not in ['20200905']:
            continue
        nclass = int(l.split()[1])
        #min_class = min(min_class, nclass)
        classes.add(nclass)
        new_lines.append(l)
    print('num_classes', len(classes), min(classes), max(classes))
    #name = os.path.basename(path)
    #root = os.path.dirname(path)
    sorted_class = sorted(classes)
    class_ind= {cls: i for (i, cls) in enumerate(sorted_class)}
    with open(path.replace('.src', ''), 'w') as wf:
        for l in new_lines:
            l_split = l.split(' ')
            l_split[1] = str(class_ind[int(l_split[1])])
            new_l = ' '.join(l_split)
            wf.write(new_l + '\n')

def batch_select_file(paths):
    for p in paths:
        print(p)
        select_file(p)

batch_select_file(glob.glob('/dockerdata/fufuyu/Projects/tools/tfrecord_tool/indexes/*_20200903to09.txt.src'))
