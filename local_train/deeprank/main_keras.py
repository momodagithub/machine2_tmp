import tensorflow as tf
from distribute_train.deeprank.model.mmoe_keras import build_mmoe_keras_model
from config.feature_column import sparse_columns, dense_columns, rank_columns, varlen_columns, position_columns
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow import feature_column as fc

# 1) 解析规范：把所有特征列 + userid(string) + 标签 放进 feature_spec
def make_feature_spec():
    feature_spec = {}
    # 单值int特征
    for c in list(sparse_columns.values()) + list(dense_columns.values()) + list(rank_columns.values()):
        feature_spec[c.name] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    # 变长int特征（序列）
    for c in varlen_columns.values():
        feature_spec[c.name] = tf.io.VarLenFeature(tf.int64)
    # 原始字符串userid（可缺省时给默认值避免解析报错）
    feature_spec['userid'] = tf.io.FixedLenFeature([], tf.string, default_value="")
    # 标签
    feature_spec['ctr_label'] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    feature_spec['cvr_label'] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    feature_spec['finish_view_label'] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    feature_spec['unlock_label'] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    return feature_spec

# 2) 解析与类型规范化：Keras的输入需要 int32（或 string）
def parse_example_fn(example_proto):
    spec = make_feature_spec()
    parsed = tf.io.parse_single_example(example_proto, spec)

    # 取出label
    labels = {
        'ctr': tf.cast(parsed.pop('ctr_label'), tf.float32),
        'cvr': tf.cast(parsed.pop('cvr_label'), tf.float32),
        'fvtr': tf.cast(parsed.pop('finish_view_label'), tf.float32),
        'unlock': tf.cast(parsed.pop('unlock_label'), tf.float32),
    }

    # Keras Inputs：单值特征转 int32，VarLen 特征保持 SparseTensor(int32)，userid 保持 string
    inputs = {}
    # 单值
    for c in list(sparse_columns.values()) + list(dense_columns.values()) + list(rank_columns.values()):
        v = parsed[c.name]
        inputs[c.name] = tf.cast(v, tf.int32)
    # 变长
    for c in varlen_columns.values():
        sp = parsed[c.name]
        inputs[c.name] = tf.SparseTensor(indices=sp.indices,
                                         values=tf.cast(sp.values, tf.int32),
                                         dense_shape=sp.dense_shape)
    # userid
    inputs['userid'] = parsed['userid']  # already string

    return inputs, labels

# 3) 构造 Dataset
def build_dataset(tfrecord_files, batch_size=512, shuffle_buffer=10000, num_parallel_calls=tf.data.AUTOTUNE):
    ds = tf.data.TFRecordDataset(tfrecord_files, compression_type=None, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(parse_example_fn, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# 4) 准备 params（需要包含 all_column：deepctr/TF feature_column 的字典，与你现有 pipeline 保持一致）
# 假设你已有 params['all_column'] 的构建逻辑，这里只示意
def build_params():
    # 将 deepctr 的 SparseFeat/VarLenSparseFeat 转成 tf.feature_column，供 fc.input_layer 使用
    def tf_col_from_sparsefeat(sf: SparseFeat):
        cat = fc.categorical_column_with_identity(key=sf.name, num_buckets=int(sf.vocabulary_size), default_value=0)
        emb = fc.embedding_column(categorical_column=cat, dimension=int(sf.embedding_dim), combiner='mean')
        return emb

    def tf_col_from_varlen(vsf: VarLenSparseFeat):
        inner: SparseFeat = vsf.sparsefeat
        cat = fc.categorical_column_with_identity(key=inner.name, num_buckets=int(inner.vocabulary_size), default_value=0)
        emb = fc.embedding_column(categorical_column=cat, dimension=int(inner.embedding_dim), combiner=vsf.combiner or 'mean')
        return emb

    all_column = {}
    # 单值类（包含 dense 离散化后按 SparseFeat 表示、rank、普通 sparse）
    for dic in (sparse_columns, dense_columns, rank_columns):
        for _k, sf in dic.items():
            col = tf_col_from_sparsefeat(sf)
            all_column[sf.name.lower()] = col
    # 序列/变长
    for _k, vsf in varlen_columns.items():
        col = tf_col_from_varlen(vsf)
        all_column[vsf.name.lower()] = col
    # 位置特征
    for sf in position_columns:
        col = tf_col_from_sparsefeat(sf)
        all_column[sf.name.lower()] = col

    return {
        'all_column': all_column
    }

# 5) 训练
def train(tfrecord_files, epochs=1, batch_size=512, lr=1e-4):
    params = build_params()
    model = build_mmoe_keras_model(params)

    # 多头二分类损失
    losses = {
        'ctr': tf.keras.losses.BinaryCrossentropy(),
        'fvtr': tf.keras.losses.BinaryCrossentropy(),
        'unlock': tf.keras.losses.BinaryCrossentropy(),
        'cvr': tf.keras.losses.BinaryCrossentropy(),
    }
    metrics = {
        'ctr': [tf.keras.metrics.AUC(name='auc')],
        'fvtr': [tf.keras.metrics.AUC(name='auc')],
        'unlock': [tf.keras.metrics.AUC(name='auc')],
        'cvr': [tf.keras.metrics.AUC(name='auc')],
    }
    loss_weights = {'ctr': 1.0, 'fvtr': 1.0, 'unlock': 1.0, 'cvr': 1.0}

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=losses, loss_weights=loss_weights, metrics=metrics)

    ds = build_dataset(tfrecord_files, batch_size=batch_size)
    model.fit(ds, epochs=epochs)

    return model

if __name__ == "__main__":
    # 示例：传入一个或多个 TFRecord 文件路径列表
    files = [
        # 'hdfs://.../part-00000.tfrecord',
        # '/path/to/local/file.tfrecord'
    ]
    model = train(files, epochs=1, batch_size=512, lr=1e-4)