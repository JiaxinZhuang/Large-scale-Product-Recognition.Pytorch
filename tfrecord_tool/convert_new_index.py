import os
import sys


def build_dict(tfr_indexes):
    d = {}
    for tfr_index in tfr_indexes:
        print("reading {}".format(tfr_index))
        tfr_name = os.path.basename(tfr_index).replace('.index', '')
        with open(tfr_index, 'r') as f:
            for line in f:
                print (line)
                file_name, shard_index, offset = line.rstrip().split('\t')
                data_name = file_name.split('/')[0]
                print (data_name, 'fooo')
                if data_name not in d:
                    d[data_name] = {}
                d[data_name][file_name] = '{}\t{}\t{}'.format(
                    tfr_name, shard_index, offset)
    print("build dict done")
    return d


def convert(index_file, d, out_index_file):
    print("write to new index file {}".format(out_index_file))
    with open(index_file, 'r') as f, open(out_index_file, 'w') as out_f:
        for line in f:
            # file_name, label = line.rstrip().split('\t')
            file_name, label = line.rstrip().split(' ')
            data_name = file_name.split('/')[0]
            if data_name not in d or file_name not in d[data_name]: continue
            tfr_string = d[data_name][file_name]
            out_f.write(tfr_string + '\t{}'.format(label) + '\n')


def main():
    if len(sys.argv) < 3:
        print(
            "python {} <index.txt> <tfr1.index> <tfr2.index> ... <new_index.txt>"
            .format(sys.argv[0]))
        sys.exit(0)

    tfr_indexes = []
    for tfr_index in sys.argv[2:-1]:
        tfr_indexes.append(tfr_index)
    d = build_dict(tfr_indexes)
    convert(sys.argv[1], d, sys.argv[-1])


if __name__ == '__main__':
    main()
