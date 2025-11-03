#coding:utf-8
import tensorflow as tf
from tensorflow import feature_column as fc

def parse_exmp(serial_exmp, feature_spec):
    feats = tf.io.parse_example(serial_exmp, features=feature_spec)
    label_ctr = feats.pop('ctr_label')
    label_cvr = feats.pop('cvr_label')
    label_vip = feats.pop('vip_label')
    label_unlock = feats.pop('unlock_label')
    label_finish = feats.pop('finish_view_label')

    labels={}
    labels['ctr_label']=label_ctr
    labels['cvr_label']=label_cvr
    labels['vip_label']=label_vip
    labels['unlock_label']=label_unlock
    labels['finish_view_label']=label_finish

    return feats, labels


def train_input_fn(filenames, shuffle_buffer_size, num_epochs, batch_size, feature_spec):
    dataset = tf.data.Dataset.list_files(filenames)\
        .shuffle(shuffle_buffer_size)\
        .repeat(num_epochs) \
        .interleave(  # Parallelize data reading
            tf.data.TFRecordDataset,
            cycle_length=batch_size,
            block_length=batch_size,
            num_parallel_calls=batch_size
        )\
        .batch(
            batch_size
        )\
        .map(  # Parallelize map transformation
            lambda x: parse_exmp(x, feature_spec),
            num_parallel_calls=batch_size
        )\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def eval_input_fn(filenames, batch_size, feature_spec):
    dataset = tf.data.Dataset.list_files(filenames) \
        .interleave(  # Parallelize data reading
            tf.data.TFRecordDataset,
            cycle_length=batch_size,
            block_length=batch_size,
            num_parallel_calls=batch_size
        ) \
        .batch(
            batch_size
        ) \
        .map(  # Parallelize map transformation
            lambda x: parse_exmp(x, feature_spec),
            num_parallel_calls=batch_size
        ) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return dataset