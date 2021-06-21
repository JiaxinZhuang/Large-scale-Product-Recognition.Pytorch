import cv2
import sys
import io
import os
import queue
import threading
import random
import tensorflow as tf
import numpy as np
import json
import argparse
import logging
from datetime import datetime as dt


def gen_imgpath(input_index, input_dir, limit, shuffle):
    fin = open(input_index, 'r')
    lines = fin.readlines()
    if limit > 0:
        lines = lines[:limit]
    if shuffle:
        random.shuffle(lines)
    samples = []
    print(lines[0])
    for line in lines:
        if '\t' in line:
            image_path = line.rstrip().split('\t')[0]
            label = line.rstrip().split('\t')[1:]
        else:
            line_s = line.rstrip().split(' ')
            if len(line_s) == 1:
                image_path = line_s[0]
                label = None
            else:
                image_path = line_s[0]
                label = line_s[1:]

        if len(input_dir) > 0:
            image_path = os.path.join(input_dir, image_path)
        samples.append((image_path, label))
    return samples


def encode_to_tfrecord(input_index, input_dir, limit, shuffle):
    samples = gen_imgpath(input_index, input_dir, limit, shuffle)
    for image_path, label in samples:
        #print ('decoding', image_path)
        if not tf.io.gfile.exists(image_path):
            print('Failed to find file: ' + image_path)
            continue
        else:
            image = open(image_path, mode='rb')

        image_raw = image.read()

        try:
            im = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        except:
            print ("openv decode:%s failed, Please check" %(image_path))
            continue

        if len(input_dir) > 0:
            image_path = image_path.replace(input_dir, '')
        if label is not None:
            label, track, camera = label
            label = np.array(int(label), dtype=np.int32).tostring()
            track = np.array(track).tostring()
            camera = np.array(int(camera), dtype=np.int32).tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        label,
                    ])),
                    'track':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        track,
                    ])),
                    'camera':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        camera,
                    ])),
                    'image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        image_raw,
                    ]))
                }))
        else:
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        image_raw,
                    ]))
                }))

        yield (image_path, example)


class TaskQueue:
    def __init__(self, input_index, input_dir, output_name, output_dir,
                 shard_size, capacity, limit, shuffle):
        self._input_index = input_index
        self._input_dir = input_dir
        self._output_name = output_name
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._limit = limit
        self._shuffle = shuffle
        self._queue = queue.Queue(maxsize=capacity)
        self._threads = []

    def produce(self):
        count = 0
        for item in encode_to_tfrecord(self._input_index, self._input_dir,
                                       self._limit, self._shuffle):
            count += 1
            if count % 100000 == 0:
                print("{}: {} processed".format(dt.now(), count))
            self._queue.put(item)
        self._queue.put(None)

    def consume(self):
        idx_file = os.path.join(self._output_dir,
                                '{}.index'.format(self._output_name))
        with open(idx_file, 'w') as idx_writer:
            count = 0
            cur_shard_size = 0
            cur_shard_idx = -1
            cur_shard_writer = None
            cur_shard_path = None
            cur_shard_offset = None
            while True:
                item = self._queue.get()
                if item is None:
                    break
                key, example = item

                if cur_shard_size == 0:
                    cur_shard_idx += 1
                    record_filename = '{0}-{1:05}.tfrecord'.format(
                        self._output_name, cur_shard_idx)
                    if cur_shard_writer is not None:
                        cur_shard_writer.close()
                    cur_shard_path = os.path.join(self._output_dir,
                                                  record_filename)
                    cur_shard_writer = tf.io.TFRecordWriter(
                        cur_shard_path)
                    cur_shard_offset = 0

                example_bytes = example.SerializeToString()
                cur_shard_writer.write(example_bytes)
                cur_shard_writer.flush()
                idx_writer.write('{}\t{}\t{}\n'.format(key, cur_shard_idx,
                                                       cur_shard_offset))
                cur_shard_offset += (len(example_bytes) + 16)

                count += 1
                cur_shard_size = (cur_shard_size + 1) % self._shard_size

            if cur_shard_writer is not None:
                cur_shard_writer.close()
            print('total examples number = {}'.format(count))
            print('total shard number = {}'.format(cur_shard_idx + 1))

    def start(self):
        reader = threading.Thread(target=self.produce, args=[])
        reader.start()
        self._threads.append(reader)
        print('reader started.')

        writer = threading.Thread(target=self.consume, args=[])
        writer.start()
        self._threads.append(writer)
        print('writer started.')

        for t in self._threads:
            t.join()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Basic statistics on tfrecord files')
    parser.add_argument('--input_index',
                        dest='input_index',
                        help='Path to the input tar dataset index file.',
                        type=str,
                        required=True)
    parser.add_argument('--input_dir',
                        dest='input_dir',
                        help='Directory of input tar dataset.',
                        type=str,
                        required=True)
    parser.add_argument('--output_name',
                        dest='output_name',
                        help='Prefix for the tfrecords.',
                        type=str,
                        required=True)
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        help='Directory for the tfrecords.',
                        type=str,
                        required=True)
    parser.add_argument('--shard_size',
                        dest='shard_size',
                        help='Number of examples per shard.',
                        type=int,
                        required=True)
    parser.add_argument('--limit',
                        dest='limit',
                        help='Number of examples encoded.',
                        type=int,
                        required=True)
    parser.add_argument('--shuffle',
                        dest='shuffle',
                        help='Whether shuffle or not..',
                        type=int,
                        required=True)
    parsed_args = parser.parse_args()
    return parsed_args


def main():
    args = parse_args()
    tq = TaskQueue(input_index=args.input_index,
                   input_dir=args.input_dir,
                   output_name=args.output_name,
                   output_dir=args.output_dir,
                   shard_size=args.shard_size,
                   limit=args.limit,
                   shuffle=(args.shuffle > 0),
                   capacity=1000)
    tq.start()


if __name__ == '__main__':
    main()
