import os
from  os.path import join as ospj
import pickle

#input_dirs = ['../trunk213_Wuhan1001to1002_Fengke0926to0927/trunk213_Wuhan1001to1002_Fengke0926to0927']
#input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina0720_refine', 
#    '/data1/sevjiang/datasets/pcb-format/huina_0719_all_refine',
#    '/data1/sevjiang/datasets/pcb-format/huina_0707_split0_train_newoffset',
#    '/data1/sevjiang/datasets/pcb-format/huina_0707_unsup_refine',
#    ]
#root = '../'
#data_list = open('data_list.txt', 'r').readlines()
#2020/3/12 sevjiang
#input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina_200210_unsup']
#input_dirs = ['/data1/sevjiang/datasets/pcb-format/cuhk03_pcb', '/data1/sevjiang/datasets/pcb-format/cuhk03_split2_pcb']
input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina_sh_poc_train_label_merge_pcb']

#input_dirs = [ospj(root, line.strip()) for line in data_list if line.strip()!=''] 
print(input_dirs)
#merge_ids = ['', '', '', '']

merge_ids = ['']*len(input_dirs)
assert(len(input_dirs) == len(merge_ids))

for merge_id_name, dataset_name in zip(merge_ids, input_dirs):

    partitions = pickle.load(open(os.path.join(dataset_name, 'partitions.pkl')))
    print partitions.keys()
    img_list = partitions['trainval_im_names']
    ids2labels = partitions['trainval_ids2labels']

    if 'test_im_names' in partitions:
        test_img_list = partitions['test_im_names']
    else:
        test_img_list = []

    train_ids = set()
    for img_name in img_list:
        train_ids.add(img_name.split('_')[0])

    test_ids = set()
    for img_name in test_img_list:
        test_ids.add(img_name.split('_')[0])
    print (ids2labels)
    if merge_id_name != '':
        merge_list = open(merge_id_name).readlines()
        merge_id_set = set()
        new_merge_list = []
        for line in merge_list:
            line = line.split()
            line =  set(line) - test_ids
            if len(line) == 0: 
                continue
            new_merge_list.append(list(line))
            merge_id_set = merge_id_set.union(line)
         
        train_ids = train_ids - merge_id_set
        print (len(train_ids), len(new_merge_list))
        ids2labels = {}
        for label, id in enumerate(sorted(list(train_ids))):
            ids2labels[int(id)] = label
        print max(ids2labels.values())
        for i, ids in enumerate(sorted(list(new_merge_list))):
            for id in ids:
                ids2labels[int(id)] = label + i + 1
    #print label + i + 1, max(ids2labels.values())

    output_name = os.path.basename(dataset_name)
    f = open('index_' + output_name + '.txt', 'w')
    for img_name in sorted(img_list):
        id = int(img_name.split('_')[0])
        f.write(output_name + '/images/' + img_name + ' ' + str(ids2labels[id]) + '\n')
    f.close()
