import os
from  os.path import join as ospj
import pickle

#input_dirs = ['../trunk213_Wuhan1001to1002_Fengke0926to0927/trunk213_Wuhan1001to1002_Fengke0926to0927']
#input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina0720_refine', 
#    '/data1/sevjiang/datasets/pcb-format/huina_0719_all_refine',
#    '/data1/sevjiang/datasets/pcb-format/huina_0707_split0_train_newoffset',
#    '/data1/sevjiang/datasets/pcb-format/huina_0707_unsup_refine',
#    ]
#input_dirs = ['../trunk213_Wuhan1001to1002_Fengke0926to0927/trunk213_Wuhan1001to1002_Fengke0926to0927']
#root = '../'
#data_list = open('data_list.txt', 'r').readlines()

#input_dirs = [ospj(root, line.strip()) for line in data_list if line.strip()!=''] 
#input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina1007_unsup_part01']
#input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina1007labeled_part01']
input_dirs = ['/data1/sevjiang/datasets/pcb-format/wanda_0811_0819_0824_0826_0827_0901_0904_0905_1001_1003_1208all_1230all_0106_baili1012_1216/']

print(input_dirs)
#merge_ids = ['', '', '', '']
merge_ids = ['', '', '', '']

#merge_ids = ['']*len(input_dirs)
assert(len(input_dirs) == len(merge_ids))

for merge_id_name, dataset_name in zip(merge_ids, input_dirs):

    partitions = pickle.load(open(os.path.join(dataset_name, 'partitions.pkl')))
    print partitions.keys()
    #img_list = partitions['trainval_im_names']
    img_list = os.listdir(os.path.join(input_dir, 'images_part'))
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
    
    output_name = os.path.basename(dataset_name) + '_part'
    f = open('index_' + output_name + '.txt', 'w')
    for img_name in sorted(sub_img_list):
        id = int(img_name.split('_')[0])
        f.write(output_name + '/images/' + img_name + ' ' + str(ids2labels[id]) + '\n')
    f.close()
