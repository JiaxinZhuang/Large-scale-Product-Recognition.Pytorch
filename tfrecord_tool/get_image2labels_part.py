import os
from  os.path import join as ospj
import pickle
import cv2

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

#input_dirs = ['/data1/sevjiang/datasets/pcb-format/wanda_0811_0819_0824_0826_0827_0901_0904_0905_1001_1003_1208all_1230all_0106_baili1012_1216']
merge_ids = ['/data1/sevjiang/datasets/pcb-format/wanda_0811_0819_0824_0826_0827_0901_0904_0905_1001_1003_1208all_1230all_0106_baili1012_1216/merged_ids_wanda_0811_0819_0824_0826_0827_0901_0904_0905_1001_1003_1208all_1230all_0106_baili1012_1216']


#2020/3/12 sevjiang
input_dirs = ['/data1/sevjiang/datasets/pcb-format/huina_200210_unsup']
merge_ids = ['']*len(input_dirs)

print(input_dirs)
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

    sub_img_list = os.listdir(os.path.join(dataset_name, 'images_part'))
    label2part = {}
    for img_name in sub_img_list:
        img_name = img_name.replace('.png', '').split('_')     
        id_name = int(img_name[0])

        if id_name not in ids2labels: continue

        part_name = img_name[3]
        label = ids2labels[id_name]
        if label not in label2part:
            label2part[label] = set()
        label2part[label].add(part_name)
    print (label2part)
    new_ids2labels = {}
    current_l = 0
    for label in label2part:
        part_names = label2part[label]
        for part_name in part_names:
            new_ids2labels[(label, part_name)] = current_l
            current_l += 1
    print (sorted(new_ids2labels.keys())) 
    output_name = os.path.basename(dataset_name)
    f = open('index_' + output_name + '_part.txt', 'w')
    for img_name in sorted(sub_img_list):
        img = cv2.imread(os.path.join(dataset_name, 'images_part', img_name))
        if img is None or img.shape[0] < 25 or img.shape[1] < 25: continue
        split_img_name = img_name.replace('.png', '').split('_')     
        id = int(split_img_name[0])
        part_name = split_img_name[3]
       
        if id not in ids2labels: continue
        #id = int(img_name.split('_')[0])
        label = ids2labels[id]
        new_label = new_ids2labels[(label, part_name)]
        print(img_name, new_label)
        f.write(output_name + '/images/' + img_name + ' ' + str(new_label) + '\n')
    f.close()
