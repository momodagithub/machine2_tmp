import sys,os
import time
import yaml
import argparse
import bisect
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Layer
from datetime import datetime
import logging
from multiprocessing import Pool
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepctr.inputs import create_embedding_matrix, embedding_lookup, varlen_embedding_lookup
from deepctr.layers.core import DNN
from tensorflow.keras import layers
from config.feature_column import sparse_columns, dense_discretized, varlen_columns, rank_discretized, dense_columns, rank_columns, position_columns
from tensorflow.keras.layers import Dense, Softmax
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import HadoopFileSystem


def str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


yaml.add_representer(str, str_presenter)


tensorboard = TensorBoard(log_dir='dz/logs', update_freq='batch')
category_features = {
    'book_id':'bid',
    'mbid':'bid',
    'prov':'prov',
    'language':'language',
    'syslangu':'language',
    'book_language':'language',
    'm_media_source':'m_media_source',
    'book_type_one':'book_type_one',
    'brand':'brand',
    'os':'os_type',
    'is_login':'is_login',
    'vip_type':'is_vip',
    'gender':'gender',
    'level_one':'level_one',
    'level_two':'level_two'
}


category_list_features = {
    'tag_set':'tagv3',
    'top_prefer_tag':'tagv3'
}
BATCH_SIZE=512
rank_features = ['mchid', 'model']
rank_counts = {feat:{} for feat in rank_features}
feat2id_map = {}
os.environ['LD_LIBRARY_PATH'] = '/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/lib/hadoop/lib/native'
feat_oos_count = {}


def date_diff(date_str1, date_str2, format_str="%Y-%m-%d %H:%M:%S"):
    if date_str1 == "" or date_str2 == "":
        return "-1"
    try:
        date1 = datetime.strptime(date_str1, format_str)
        date2 = datetime.strptime(date_str2, format_str)

        delta = date1 - date2
        return delta.days
    except Exception as e:
        return -1


def cat_list_2_id_list(cat_value, emb_name, cat_size, cat_name, max_size, rtype="array"):
    cat_list = str(cat_value).replace("_", ",").split(',')
    if emb_name=="gap_day" or emb_name == "play_gap":
        res = [cat_2_id(str(min(int(cat), 190) if cat else cat), emb_name, cat_size, cat_name) for cat in cat_list]
    else:
        res = [cat_2_id(cat, emb_name, cat_size, cat_name) for cat in cat_list]
    res = res[:max_size] + [0] * (max_size - len(res))
    if rtype == "array":
        return np.array(res)
    else:
        return res


def cat_2_id(cat_value, emb_name, cat_size, cat_name):
    if emb_name not in feat2id_map:
        feat2id_map[emb_name] = {}
    if (cat_value != cat_value) or (not cat_value.strip()):
        return 0
    if cat_value not in feat2id_map[emb_name]:
        max_size = len(feat2id_map[emb_name])
        if max_size < cat_size - 1:
            feat2id_map[emb_name][cat_value] = max_size + 1
        else:
            feat_oos_count[emb_name] = feat_oos_count.get(emb_name, 0) + 1
            print(f"{cat_name}:{cat_value} out of range, size={cat_size}")
            print(feat2id_map[emb_name])
            print('----------------------')
            feat2id_map[emb_name][cat_value] = 0
    return feat2id_map[emb_name][cat_value]


def load_yaml(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f'---------------load config success, size:{len(config)}-----------------------')
            return config
    except Exception as e:
        print(f'{config_path} load failed, will load an empty file, pls check is it ok ...')
        return {}


def save_feature_dic_yaml():
    pass


class BatchLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        auc = logs.get('auc')
        print(f"Batch {batch}, Loss: {loss}, AUC: {auc}")


class MutiTaskBatchLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        print(f"Batch {batch}:", end=" ")
        for task in ['ctr', 'cvr']:
            task_loss = logs.get(f'{task}_loss', 'N/A')
            task_auc = logs.get(f'{task}_auc', 'N/A')
            print(f"{task.upper()} - Loss: {task_loss}, AUC: {task_auc};", end=" ")
        print('')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_raw_data(raw_data_path, columns, chunksize=128 * 20):
    data_iter = pd.read_csv(raw_data_path, sep='\t', names=columns,  na_values=['\\N'], quoting=3, chunksize=chunksize, iterator=True)
    return data_iter


def extrac_tag(x):
    try:
        return ','.join(map(lambda tag: tag.split('_', 1)[1], x.split(',')))
    except Exception as e:
        return 'unknown'


def trans_data(chunk):
    try:
        chunk.rename(columns=cname2fname, inplace=True)
        condition = ~((chunk['mtime_gap'] >= 0) & (chunk['mtime_gap'] < 60) & (chunk['ctr_label'] == 0) & (
                        chunk['cvr_label'] == 0) & (chunk['unlock_label'] == 0) & (chunk['play_ratio'] < 0.1))
        chunk = chunk[condition].reset_index(drop=True)
        chunk['finish_view_label'] = (chunk['play_ratio'] > 0.9).astype(int)

        neg_condition = (
                (chunk['ctr_label'] == 0) &
                (chunk['cvr_label'] == 0) &
                (chunk['finish_view_label'] == 0) &
                (chunk['unlock_label'] == 0) &
                (chunk['play_ratio'] < 0.1)
        )

        neg_samples = chunk[neg_condition]
        non_neg_samples = chunk[~neg_condition]
        sampled_neg = neg_samples.sample(frac=0.5, random_state=42)
        chunk = pd.concat([non_neg_samples, sampled_neg], ignore_index=True)
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

        chunk['vip_label'] = chunk['vip_type'].replace('', 0).fillna(0).astype(int)
        chunk['book_pos'] = chunk['book_pos'].fillna(60).astype(int).apply(lambda x: min(x, 60))
        chunk['hour'] = chunk['reqtime'].apply(
            lambda x: x[11:13] if isinstance(x, str) and len(x) >= 13 else -1).astype(int)
        chunk['gap_day'] = '0'
        for _, sparse_data in dense_columns.items():
            category_feat, emb_name, feat_size = sparse_data.name, sparse_data.embedding_name, sparse_data.vocabulary_size
            if category_feat not in feature_names:
                continue
            discretized = dense_discretized.get(category_feat)
            chunk[category_feat] = chunk[category_feat].fillna(0.0).astype(float).apply(
                lambda x: bisect.bisect_right(discretized, x))
    except Exception as e:
        print("info===", type(chunk), "\tcolumns=", chunk.columns)
        logging.exception("process error")
        print('-' * 30)
    return 0, chunk


def process_chunk(chunk):
    for _, sparse_data in sparse_columns.items():
        category_feat, emb_name, feat_size = sparse_data.name, sparse_data.embedding_name, sparse_data.vocabulary_size
        if category_feat not in feature_names:
            continue
        chunk[category_feat] = chunk[category_feat].fillna("unknown").astype(str).apply(lambda x: cat_2_id(str(x).lower(), emb_name, feat_size, category_feat))

    for _, varlen_data in varlen_columns.items():
        category_feat, emb_name, max_size, feat_size = varlen_data.name, varlen_data.embedding_name, varlen_data.maxlen, varlen_data.vocabulary_size
        if category_feat not in feature_names:
            continue
        chunk[category_feat] = chunk[category_feat].apply(lambda x: cat_list_2_id_list(x, emb_name, feat_size, category_feat, max_size))

    for feat in rank_features:
        counts = chunk[chunk[feat] != ''][feat].value_counts()
        for key, value in counts.items():
            rank_counts[feat][key] = rank_counts[feat].get(key, 0) + value

        old_feat_count = feat2id_map.get(feat, {})
        chunk[feat] = chunk[feat].map(old_feat_count).fillna(0).astype(int)

    if len(chunk) != BATCH_SIZE:
        return 1, chunk
    feats = {name: np.stack(chunk[name].values) for name in feature_names}
    labels = {label: np.stack(chunk[label].values) for label in label_names}
    return 0, (feats, labels)


def data_generator(data_iter, num_processes=25):
    tmp_data = None
    with Pool(num_processes) as pool:
        for chunk in data_iter:
            chunk = chunk.sample(frac=1.0)
            chunk_size = len(chunk)
            rows_per_process = chunk_size // num_processes
            ranges = [(i * rows_per_process, (i + 1) * rows_per_process) if i < num_processes - 1
                      else (i * rows_per_process, chunk_size) for i in range(num_processes)]
            results = pool.map(trans_data, [chunk.iloc[start:end, :] for start, end in ranges])
            for flag0, data in results:
                if flag0 != 0:
                    continue
                try:
                    flag, result = process_chunk(data)
                except Exception as e:
                    logging.exception("process_chunk error")
                    continue
                if flag != 0:
                    if tmp_data is None:
                        tmp_data = result
                        continue
                    else:
                        tmp_data = pd.concat([tmp_data, result], copy=False)
                        if len(tmp_data) < BATCH_SIZE:
                            continue
                        else:
                            result = tmp_data[:BATCH_SIZE]
                            tmp_data = tmp_data[BATCH_SIZE:]

                            feats = {name: np.stack(result[name].values) for name in feature_names}
                            labels = {label: np.stack(result[label].values) for label in label_names}
                            result = feats, labels
                yield result


def get_dataset_from_generator(train_sample_gen):
    dataset = tf.data.Dataset.from_generator(lambda :train_sample_gen, output_signature=(feat_sig, label_sig))
    dataset = dataset.prefetch(2).batch(batch_size=1)
    return dataset


def get_attention(query_input, key_input, att_emb_size=128):
    query_input0 = tf.expand_dims(query_input, axis=1)
    query = Dense(att_emb_size, name="query_dense")(query_input0)
    keys = Dense(att_emb_size, name="key_dense")(key_input)
    values = Dense(att_emb_size, name="value_dense")(key_input)

    scores = tf.matmul(query, keys, transpose_b=True)
    scores = scores / tf.sqrt(tf.cast(att_emb_size, tf.float32))
    attention_weights = Softmax(axis=-1, name="attention_softmax")(scores)

    weighted_sum = tf.matmul(attention_weights, values)
    result = tf.reshape(weighted_sum, (-1, att_emb_size))
    return result


class Expert(Layer):
    def __init__(self, expert_hidden_units, output_units, name="expert"):
        super(Expert, self).__init__(name=name)
        self.expert_layers = []
        for units in expert_hidden_units:
            self.expert_layers.append(Dense(units, activation='relu'))
        self.output_layer = Dense(output_units, activation='relu')

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.expert_layers:
            x = layer(x)
        return self.output_layer(x)


class GatingNetwork(Layer):
    def __init__(self, num_experts, name="gating"):
        super(GatingNetwork, self).__init__(name=name)
        self.gate = Dense(num_experts, activation='softmax')

    def call(self, inputs, **kwargs):
        return self.gate(inputs)


class MultiExpertLayer(Layer):
    def __init__(self, num_experts, expert_hidden_units, expert_output_units, name="multi_expert"):
        super(MultiExpertLayer, self).__init__(name=name)
        self.experts = [
            Expert(expert_hidden_units, expert_output_units, name=f"expert_{i}")
            for i in range(num_experts)
        ]

    def call(self, inputs, **kwargs):
        outputs = [expert(inputs) for expert in self.experts]
        return tf.stack(outputs, axis=1)


from tensorflow import feature_column as fc


from config.feature_column import *


def model_fn(features, labels, mode, params):
    l2_reg_embedding = params.get("l2_reg_embedding", 0.00001)
    seed = params.get("seed", 1024)
    seq_mask_zero = params.get("seq_mask_zero", True)

    feature_columns_dic = {**sparse_columns, **dense_columns, **rank_columns, **varlen_columns}
    feature_columns = [feature_columns_dic[key] for key in sorted(feature_columns_dic)]
    feat_columns=feature_columns
    pos_columns =position_columns

    query_columns = ["book_id", "gender", "gap_day"]
    seq_columns = ['userWatchBookListLatestBookId', 'userWatchBookListLatestGapDay', 'userWatchBookListLatestGender'
                           ,'userFollowBookListLatestBookId', 'userFollowBookListLatestGapDay', 'userFollowBookListLatestGender'
                           ,'userAttributeBookListLatestBookId', 'userAttributeBookListLatestGapDay', 'userAttributeBookListLatestGender'
                           ,'userClickBookListLatestBookId', 'userClickBookListLatestGapDay', 'userClickBookListLatestGender'
                           ,'userFullWatchBookListLatestBookId', 'userFullWatchBookListLatestGapDay', 'userFullWatchBookListLatestGender'
                           ,'userBuyBookListLatestBookId', 'userBuyBookListLatestGapDay', 'userBuyBookListLatestGender'
                           ,'userOnlyUnFollowBookListLatestBookId', 'userOnlyUnFollowBookListLatestGapDay', 'userOnlyUnFollowBookListLatestGender'
                           ,'userFreeFullWatchBookListLatestBookId', 'userFreeFullWatchBookListLatestGapDay', 'userFreeFullWatchBookListLatestGender'
                           ,'userLastWatchBookListBookId', 'userLastWatchBookListGapDay', 'userLastWatchBookListGender'
                           ,'userWatchBookListGt5EpdBookId', 'userWatchBookListGt5EpdGapDay', 'userWatchBookListGt5EpdGender'
                           ,'userWatchBookListGt10EpdBookId', 'userWatchBookListGt10EpdGapDay', 'userWatchBookListGt10EpdGender'
                           ,'userFullWatchBookListLatestCheckedBookId', 'userFullWatchBookListLatestCheckedGapDay', 'userFullWatchBookListLatestCheckedGender']
    gate_columns = ["language", "book_language", "vip_type", "shelf_days", "register_min", "click_pv_7days", "prov", "page", "ad_gap_min", "uVipLabel", "activeDay", "activeDays7d"]

    learning_rate = params.get("learning_rate", 0.0001)
    global_steps = tf.compat.v1.train.get_or_create_global_step()

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feat_columns)) if feat_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feat_columns)) if feat_columns else []

    feature_columns_tf = params['all_column']

    emb_cache = {}

    def get_cached_emb(name):
        key = name.lower()
        if key in emb_cache:
            return emb_cache[key]
        this_emb = fc.input_layer(features, [feature_columns_tf[key]])
        emb_cache[key] = this_emb
        return this_emb

    sparse_embedding_list0 = [get_cached_emb(f.name) for f in sparse_feature_columns]
    gate_embedding_list0 = [get_cached_emb(n) for n in gate_columns]

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    for fc0 in varlen_sparse_feature_columns:
        feature_name = fc0.name
        if feature_name in seq_columns:
            history_feature_columns.append(fc0)
        else:
            sparse_varlen_feature_columns.append(fc0)

    query_emb_base = [get_cached_emb(n) for n in query_columns]
    repeats = int(len(seq_columns)/len(query_columns)) if len(query_columns) > 0 else 0
    query_emb_list0 = query_emb_base * repeats

    keys_emb_list = [get_cached_emb(n) for n in seq_columns]

    if mode == tf.estimator.ModeKeys.PREDICT:
        sparse_embedding_list = [tf.reshape(emb, shape=[-1, 16]) for emb in sparse_embedding_list0]
        query_emb_list = [tf.reshape(emb, shape=[-1, 16]) for emb in query_emb_list0]
        gate_embedding_list = [tf.reshape(emb, shape=[-1, 16]) for emb in gate_embedding_list0]
    else:
        sparse_embedding_list = sparse_embedding_list0
        query_emb_list = query_emb_list0
        gate_embedding_list = gate_embedding_list0

    query_input = tf.concat(query_emb_list, axis=-1)
    key_input = tf.concat(keys_emb_list, axis=-1)
    attention_emb = get_attention(query_input, key_input)

    sequence_embed_list = [get_cached_emb(f.name) for f in sparse_varlen_feature_columns]

    inputs = tf.concat(sparse_embedding_list+sequence_embed_list, axis=1)
    gate_inputs = tf.concat(gate_embedding_list, axis=1)

    feat_num = len(sparse_feature_columns + sparse_varlen_feature_columns)
    inputs_reshape = tf.reshape(inputs, (-1, feat_num, 16))

    share_gate_0 = DNN(hidden_units=(512,))(tf.stop_gradient(gate_inputs))
    share_gate_out = Dense(feat_num, use_bias=False, activation="sigmoid", name="share_gate_out")(share_gate_0)
    share_input_gated = tf.reshape(tf.expand_dims(share_gate_out, 2) * inputs_reshape, (-1, feat_num * 16))
    share_input = tf.concat([share_input_gated, attention_emb], axis=1)

    ctr_gate_0 = DNN(hidden_units=(512,))(tf.stop_gradient(gate_inputs))
    ctr_gate_out = Dense(feat_num, use_bias=False, activation="sigmoid", name="ctr_gate_out")(ctr_gate_0)
    ctr_input_gated = tf.reshape(tf.expand_dims(ctr_gate_out, 2) * inputs_reshape, (-1, feat_num * 16))
    ctr_input = tf.concat([ctr_input_gated, attention_emb], axis=1)

    finish_gate_0 = DNN(hidden_units=(512,))(tf.stop_gradient(gate_inputs))
    finish_gate_out = Dense(feat_num, use_bias=False, activation="sigmoid", name="finish_gate_out")(finish_gate_0)
    finish_input_gated = tf.reshape(tf.expand_dims(finish_gate_out, 2) * inputs_reshape, (-1, feat_num * 16))
    finish_input = tf.concat([finish_input_gated, tf.stop_gradient(attention_emb)], axis=1)

    unlock_gate_0 = DNN(hidden_units=(512,))(tf.stop_gradient(gate_inputs))
    unlock_gate_out = Dense(feat_num, use_bias=False, activation="sigmoid", name="unlock_gate_out")(unlock_gate_0)
    unlock_input_gated = tf.reshape(tf.expand_dims(unlock_gate_out, 2) * inputs_reshape, (-1, feat_num * 16))
    unlock_input = tf.concat([unlock_input_gated, tf.stop_gradient(attention_emb)], axis=1)

    cvr_gate_0 = DNN(hidden_units=(512,))(tf.stop_gradient(gate_inputs))
    cvr_gate_out = Dense(feat_num, use_bias=False, activation="sigmoid", name="cvr_gate_out")(cvr_gate_0)
    cvr_input_gated = tf.reshape(tf.expand_dims(cvr_gate_out, 2) * inputs_reshape, (-1, feat_num * 16))
    cvr_input = tf.concat([cvr_input_gated, tf.stop_gradient(attention_emb)], axis=1)

    shared_out = MultiExpertLayer(num_experts=3, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="shared_experts")(share_input)
    ctr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="ctr_experts")(ctr_input)
    ctr_expert_gate = GatingNetwork(4, name="gateCtr")(ctr_input)
    experts_for_ctr = tf.concat([shared_out, ctr_out], axis=1)
    ctr_emb = tf.reduce_sum(tf.expand_dims(ctr_expert_gate, -1) * experts_for_ctr, axis=1)

    fvtr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="fvtr_experts")(finish_input)
    fvtr_expert_gate = GatingNetwork(4, name="gateCtr")(finish_input)
    experts_for_ctr = tf.concat([shared_out, fvtr_out], axis=1)
    finish_emb = tf.reduce_sum(tf.expand_dims(fvtr_expert_gate, -1) * experts_for_ctr, axis=1)

    utr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="utr_experts")(unlock_input)
    utr_expert_gate = GatingNetwork(4, name="gateCtr")(unlock_input)
    experts_for_ctr = tf.concat([shared_out, utr_out], axis=1)
    unlock_emb = tf.reduce_sum(tf.expand_dims(utr_expert_gate, -1) * experts_for_ctr, axis=1)

    cvr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="cvr_experts")(cvr_input)
    cvr_expert_gate = GatingNetwork(4, name="gateCtr")(cvr_input)
    experts_for_ctr = tf.concat([shared_out, cvr_out], axis=1)
    cvr_emb = tf.reduce_sum(tf.expand_dims(cvr_expert_gate, -1) * experts_for_ctr, axis=1)

    ctr = layers.Dense(1, activation="sigmoid", use_bias=False, name='ctr_out')(ctr_emb)
    fvtr = layers.Dense(1, activation="sigmoid", use_bias=False, name='finish_out')(finish_emb)
    unlock = layers.Dense(1, activation="sigmoid", use_bias=False, name='unlock_out')(unlock_emb)
    cvr = layers.Dense(1, activation="sigmoid", use_bias=False, name='cvr_out')(cvr_emb)

    predict_result = {"ctr": ctr, "fvtr": fvtr, "unlock": unlock, "cvr": cvr}
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predict_result)
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predict_result,
            export_outputs=export_outputs)

    pos_embedding_list = [get_cached_emb(f.name) for f in pos_columns]
    pos_inputs = tf.concat(pos_embedding_list, axis=1)
    pos_top_emb = DNN(hidden_units=(512, 64))(pos_inputs)
    pos_logit = layers.Dense(1, activation="sigmoid", use_bias=False, name="pos_logit")(pos_top_emb)

    ctr = ctr * pos_logit
    ctfvtr = tf.multiply(ctr, fvtr)
    ctunlock = tf.multiply(ctr, unlock)
    ctcvr = tf.multiply(ctr, cvr)

    ctr_label = tf.reshape(tf.cast(labels['ctr_label'], tf.float32), (-1, 1))
    cvr_label = tf.reshape(tf.cast(labels['cvr_label'], tf.float32), (-1, 1))
    finish_view_label = tf.reshape(tf.cast(labels['finish_view_label'], tf.float32), (-1, 1))
    unlock_label = tf.reshape(tf.cast(labels['unlock_label'], tf.float32), (-1, 1))
    vip_type = tf.reshape(tf.cast(labels['vip_label'], tf.int8), (-1, 1))
    one = tf.fill(tf.shape(ctr_label), 1.0)
    zero = tf.fill(tf.shape(ctr_label), 0.0)

    train_cvr = tf.where(vip_type > 0, zero, one)

    ctr_loss = tf.compat.v1.losses.log_loss(labels=ctr_label, predictions=ctr)
    fvtr_loss = tf.compat.v1.losses.log_loss(labels=finish_view_label, predictions=ctfvtr)
    unlock_loss = tf.compat.v1.losses.log_loss(labels=unlock_label, predictions=ctunlock)
    ctcvr_loss = tf.compat.v1.losses.log_loss(labels=cvr_label, predictions=ctcvr, weights=train_cvr)
    loss = ctr_loss + fvtr_loss + unlock_loss + ctcvr_loss

    tf.compat.v1.summary.scalar(f'train/ctr_loss', ctr_loss)
    tf.compat.v1.summary.scalar(f'train/fvtr_loss', fvtr_loss)
    tf.compat.v1.summary.scalar(f'train/unlock_loss', unlock_loss)
    tf.compat.v1.summary.scalar(f'train/ctcvr_loss', ctcvr_loss)
    tf.compat.v1.summary.scalar(f'train/loss', loss)
    summary_op = tf.compat.v1.summary.merge_all()
    summary_hook = tf.compat.v1.train.SummarySaverHook(
        save_steps=200,
        output_dir=params["model_dir"],
        summary_op=summary_op)

    ctr_auc = tf.compat.v1.metrics.auc(ctr_label, ctr, num_thresholds=1001)
    fvtr_auc = tf.compat.v1.metrics.auc(finish_view_label, fvtr, num_thresholds=1001)
    unlock_auc = tf.compat.v1.metrics.auc(unlock_label, unlock, num_thresholds=1001)
    cvr_auc = tf.compat.v1.metrics.auc(cvr_label, cvr, weights=train_cvr, num_thresholds=1001)
    eval_metric_ops = {"eval/ctr_auc": ctr_auc, "fvtr_auc": fvtr_auc, "unlock_auc": unlock_auc, "cvr_auc": cvr_auc,
                       "eval/ctr_loss": tf.compat.v1.metrics.mean(ctr_loss),
                       "eval/fvtr_loss": tf.compat.v1.metrics.mean(fvtr_loss),
                       "eval/unlock_loss": tf.compat.v1.metrics.mean(unlock_loss),
                       "eval/ctcvr_loss": tf.compat.v1.metrics.mean(ctcvr_loss),
                       "eval/loss": tf.compat.v1.metrics.mean(loss)}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

        log_tensor_dict = {'global_steps':global_steps,
                           'train/loss': loss,
                           "train/ctr_auc": ctr_auc[1],
                           "train/fvtr_auc": fvtr_auc[1],
                           "train/unlock_auc": unlock_auc[1],
                           "train/cvr_auc": cvr_auc[1],
                           "train/ctr_loss": ctr_loss,
                           "train/fvtr_loss": fvtr_loss,
                           "train/unlock_loss": unlock_loss,
                           "train/ctcvr_loss": ctcvr_loss,
                           }

        logging_hook = tf.compat.v1.train.LoggingTensorHook(log_tensor_dict, every_n_iter=200, at_end=False)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=[ logging_hook],
            predictions=predict_result,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=eval_metric_ops)


def serving_input_fn():
    feature_map = {}
    for feat in list(sparse_columns.values())+list(dense_columns.values())+list(rank_columns.values()):
        feature_map[feat.name] = tf.Variable(tf.TensorShape([]), dtype=tf.int32, name=feat.name)
    for _, feat in varlen_columns.items():
        feature_map[feat.name] = tf.Variable(tf.TensorShape([feat.maxlen]), dtype=tf.int32, name=feat.name)
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)


def get_data_from_hdfs(mdate):
    column_list = [fname.lower() for fname in feature_names]+['ctr_label', 'unlock_label', 'cvr_label', 'play_ratio', 'mtime_gap', 'reqtime']
    column_list.remove("gap_day")
    for hour in range(24):
        str_hour = str(hour) if hour >= 10 else '0'+str(hour)
        directory='/user/hive/warehouse/al.db/reco_fine_ranking_samples_label_hi/dt='+mdate+"/hour="+str_hour+'/scene_id=ovs_vd_cnxh_rec'
        parquet_files = []
        cnt = 0
        while len(parquet_files) < 1:
            if cnt > 0:
                time.sleep(600)
            try:
                file_entries = hdfs.get_file_info(pa.fs.FileSelector(directory, recursive=True))
            except Exception as e:
                print(f"get_file_info error: {e}")
                cnt += 1
                continue

            parquet_files = [entry.path for entry in file_entries if entry.type == pa.fs.FileType.File]
            nofiles = [entry.path for entry in file_entries if entry.type != pa.fs.FileType.File]
            cnt += 1
            print('hour==', str_hour, '\tcnt==', cnt, '\tfiles ==', len(parquet_files), '\tdirectory_num==', len(nofiles))

        for file_path in parquet_files:
            try:
                with hdfs.open_input_file(file_path) as f:
                    parquet_file = pq.ParquetFile(f)
                    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE * 20, columns=column_list):
                        df = batch.to_pandas()
                        yield df
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")


def dump_feat_map():
    for name, feat2id_dic in rank_counts.items():
        sorted_items = sorted(feat2id_dic.items(), key=lambda x: x[1], reverse=True)
        ranked_data = {key: rank + 1 for rank, (key, value) in enumerate(sorted_items)}
        filtered_data = {key: value for key, value in ranked_data.items() if value <= rank_discretized[name]}
        feat2id_map[name + "_new"] = filtered_data

    for feat_name, discretized in dense_discretized.items():
        feat2id_map[feat_name] = ','.join(list(map(lambda x: str(x), discretized)))

    feat2id_map['featName2embName'] = {feature.name: feature.embedding_name for feature in feature_columns + position_columns}
    feat2id_map['bucketFeatureSet'] = [k for k, v in dense_discretized.items()]
    feat2id_map['categoryFeatureSet'] = list(set([feature.name for feature in list(sparse_columns.values()) + list(
        rank_columns.values()) + position_columns]) - set(feat2id_map['bucketFeatureSet']))
    feat2id_map['sequenceFeatureSet'] = [feature.name for feature in varlen_columns.values()]
    feat2id_map['seqSizeMap'] = {feature.name: feature.maxlen for feature in varlen_columns.values()}
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(feat2id_map, f, default_flow_style=False, allow_unicode=True)

    if len(feat_oos_count) > 0:
        print('----------------以下特征空间已不足, 存在id映射冲突, 需要关注--------------------')
        print(feat_oos_count)


