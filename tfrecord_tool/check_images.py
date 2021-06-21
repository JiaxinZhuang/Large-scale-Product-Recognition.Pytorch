import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import glob
import tensorflow as tf

def check_file(txtpath, root='/dockerdata/fufuyu/wanda_reid_data'):
    with open(txtpath) as rf:
        lines = rf.read().splitlines()
    new_lines = []
    for idl, l in enumerate(lines):
        if idl % 10000 == 0:
            print(idl)
        filename = l.split()[0]
        filepath = os.path.join(root, filename)
        #image = open(filepath, mode='rb')
        #image_raw = image.read()
        #im = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        try:
            image = open(filepath, mode='rb')
            image_raw = image.read()
            im = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        except:
            print ("openv decode:%s failed, Please check" %(filepath), idl)
            continue
        if im is None:
            print ("openv decode:%s None, Please check" %(filepath), idl)
            continue
        if 0 in im.shape:
            print('0 shape', filepath, idl)
            continue
        new_lines.append(l)


def tf_check_file(txtpath, root='/dockerdata/fufuyu/wanda_reid_data'):
    with open(txtpath) as rf:
        lines = rf.read().splitlines()
    new_lines = []
    bad_lines = []
    for idl, l in enumerate(lines):
        if idl % 10000 == 0:
            print(idl)
        filename = l.split()[0]
        filepath = os.path.join(root, filename)
        #image = open(filepath, mode='rb')
        #image_raw = image.read()
        #im = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        try:
            with tf.device('/cpu'):
                image = tf.io.read_file(filepath)
                img_reader = tf.io.decode_jpeg(image, channels=3, dct_method='INTEGER_ACCURATE')
        except:
            print ("tf decode:%s failed, Please check" %(filepath), idl)
            continue
        new_lines.append(l)
    with open(txtpath+'.check', 'w') as wf:
        for l in new_lines:
            wf.write(l+'\n')
    with open('./badimg.txt', 'a') as af:
        for l in bad_lines:
            af.write(l+'\n')

if __name__ == '__main__':
    for p in glob.glob('/dockerdata/fufuyu/Projects/tools/tfrecord_tool/indexes/index_*20200903to09.txt'):
        if 'dongguan' not in p:
            continue
        print(p)
        tf_check_file(p)
