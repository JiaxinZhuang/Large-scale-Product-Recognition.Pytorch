import struct
from .yt_example_pb2 import *

def getImageData(tf_file, offset):
    with open(tf_file, 'rb') as tf:
        tf.seek(int(offset))
        pb_len_bytes = tf.read(8)
        if len(pb_len_bytes) < 8:
            print("read pb_len_bytes err,len(pb_len_bytes)=" +
                  str(len(pb_len_bytes)))
            return None

        pb_len = struct.unpack('L', pb_len_bytes)[0]

        len_crc_bytes = tf.read(4)
        if len(len_crc_bytes) < 4:
            print("read len_crc_bytes err,len(len_crc_bytes)=" +
                  str(len(len_crc_bytes)))
            return None

        len_crc = struct.unpack('I', len_crc_bytes)[0]

        pb_data = tf.read(pb_len)
        if len(pb_data) < pb_len:
            print("read pb_data err,len(pb_data)=" + str(len(pb_data)))
            return None

        data_crc_bytes = tf.read(4)
        if len(data_crc_bytes) < 4:
            print("read data_crc_bytes err,len(data_crc_bytes)=" +
                  str(len(data_crc_bytes)))
            return None

        data_crc = struct.unpack('I', data_crc_bytes)[0]

        example = yt_example_pb2.Example()
        example.ParseFromString(pb_data)

        image_data_feature = example.features.feature.get("image")
        label_feature = example.features.feature.get("label")

        if image_data_feature:
            image_data = image_data_feature.bytes_list.value[0]
            return image_data
