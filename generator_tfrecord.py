from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from collections import namedtuple, OrderedDict
from dog_name_pixel_value import class_text_to_int
import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util



DATA_BASE_PATH = r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\fine_tuning_practice' + '\\'
image_dir =r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\dog_images'






def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


"""
def console_to_str(s):
    try:
        return s.decode(sys.__stdout__.encoding)
    except UnicodeDecodeError:
        return s.decode('CP949')  # utf-8로 되어 있는 부분을 CP949로 바꾸어준다.
                #

def native_str(s, replace=False):
    if isinstance(s, bytes):
        return s.decode('utf-16', 'replace' if replace else 'strict')
    return s
"""


def create_tf_example(group, path):

    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    #with tf.io.gfile.GFile(path + '\\' + getattr(group, 'filename'), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf-8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


csv_path = r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\csv_file'

csv_folder = os.listdir(csv_path)

print(csv_folder)
for csv in csv_folder:

    #if csv == 'train_labels.csv':

     #   writer = tf.io.TFRecordWriter(DATA_BASE_PATH + 'tfrecord_file\\' +csv[:-4] + '.record')

    #else:
     #   writer = tf.io.TFRecordWriter(DATA_BASE_PATH + 'tfrecord_file\\' +csv[:-4] + '.record')

    writer = tf.io.TFRecordWriter(DATA_BASE_PATH+ 'tfrecord_file\\' + csv[:-4] + '.record')
    path = r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\dog_images'
    examples = pd.read_csv(csv_path + '\\' + csv)
    grouped = split(examples, 'filename')
    #print(grouped)
    for group in range(1, len(grouped)):

        #if group.filename == '%s.jpg':
        #    continue


        tf_example = create_tf_example(grouped[group], path)
        writer.write(tf_example.SerializeToString())

    writer.close()

    #if csv == 'train_labels.csv':

    output_path = os.path.join(os.getcwd(), DATA_BASE_PATH + 'tfrecord_file\\' +csv[:-4] + '.record')
    print('Successfully created the TFRecords: {}'.format(DATA_BASE_PATH + 'tfrecord_file\\' +csv[:-4] + '.record'))
    #else:
     #   output_path = os.path.join(os.getcwd(), DATA_BASE_PATH + 'tfrecord_file\\' +csv[:-4] + '.record')
      #  print('Successfully created the TFRecords: {}'.format(DATA_BASE_PATH + 'tfrecord_file\\' +csv[:-4] + '.record'))