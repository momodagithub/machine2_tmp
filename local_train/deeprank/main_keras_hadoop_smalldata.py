import tensorflow as tf
#from model.mmoe_keras import build_mmoe_keras_model
from model.mmoe_keras_notdeepctr import build_mmoe_keras_model
from config.feature_column import sparse_columns, dense_columns, rank_columns, varlen_columns, position_columns, int_sparse_feature
from tensorflow import feature_column as fc
#tf.get_logger().setLevel('ERROR')               # 去掉 TF 自带 INFO/WARNING
#tf.keras.utils.disable_interactive_logging()    # 关掉 1/Unknown 那种进度条
#tf.enable_eager_execution()


        # keras.backend
# 1) 构建与 feature_process 部分一致的 all_column（不包含 userid）
def build_model_columns_like_feature_process():
    feature_columns = {}

    # dense 特征：已离散化，identity + embedding
    for _, v in dense_columns.items():
        f_name = v.name
        f_embedding = fc.embedding_column(
            fc.categorical_column_with_identity(
                f_name, num_buckets=v.vocabulary_size
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding

    # sparse 特征：区分整型与字符串，userid 由模型内部处理，这里跳过
    for _, v in sparse_columns.items():
        f_name = v.name
        if f_name.lower() == 'userid':
            continue
        f_embedding = fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name,
                hash_bucket_size=v.vocabulary_size,
                dtype=tf.int64 if f_name.lower() in int_sparse_feature else tf.string
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding

    # varlen/序列特征：统一按 int64 哈希 + mean combiner
    for _, v in varlen_columns.items():
        f_name = v.name
        f_embedding = fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.int64
            ),
            dimension=16,
            combiner='mean'
        )
        feature_columns[f_name.lower()] = f_embedding

    # rank 特征：按字符串哈希
    for _, v in rank_columns.items():
        f_name = v.name
        f_embedding = fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.string
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding

    # position 特征：按 int64 哈希
    for v in position_columns:
        f_name = v.name
        f_embedding = fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.int64
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding

    return feature_columns


# 2) 解析规范：把所有特征列 + userid(string) + 标签 放进 feature_spec
def make_feature_spec():
    feature_spec = {}
    # 单值特征：区分 int 和 string 类型
    for c in list(sparse_columns.values()):
        if c.name.lower() in int_sparse_feature:
            feature_spec[c.name] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        else:
            feature_spec[c.name] = tf.io.FixedLenFeature([], tf.string, default_value="")
    
    # 单值dense特征（全部是int）
    for c in list(dense_columns.values()):
        feature_spec[c.name] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    
    # rank特征（全部是string）
    for c in list(rank_columns.values()):
        feature_spec[c.name] = tf.io.FixedLenFeature([], tf.string, default_value="")
    
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


# 3) 解析与类型规范化：Keras的输入需要 int32（或 string）
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

    # Keras Inputs：根据特征类型进行处理
    inputs = {}
    
    # sparse特征：区分int和string类型
    for c in sparse_columns.values():
        if c.name in parsed:
            if c.name.lower() in int_sparse_feature:
                inputs[c.name] = tf.cast(parsed[c.name], tf.int32)
            else:
                inputs[c.name] = parsed[c.name]  # 保持string类型
    
    # dense特征：全部转为int32
    for c in dense_columns.values():
        if c.name in parsed:
            inputs[c.name] = tf.cast(parsed[c.name], tf.int32)
    
    # rank特征：保持string类型
    for c in rank_columns.values():
        if c.name in parsed:
            inputs[c.name] = parsed[c.name]  # 保持string类型
    
    # 变长特征：转为SparseTensor(int32)
    for c in varlen_columns.values():
        if c.name in parsed:
            sp = parsed[c.name]
            inputs[c.name] = tf.SparseTensor(indices=sp.indices,
                                           values=tf.cast(sp.values, tf.int32),
                                           dense_shape=sp.dense_shape)
    
    # userid：保持string类型
    if 'userid' in parsed:
        inputs['userid'] = parsed['userid']

    return inputs, labels


# 4) 构造 Dataset
def build_dataset(tfrecord_files, batch_size=1024, shuffle_buffer=10000, ):
    #ds = tf.data.TFRecordDataset(tfrecord_files, compression_type=None)
    #ds = ds.shuffle(shuffle_buffer)
    #ds = ds.map(parse_example_fn)


    # 1. 多文件并行读取 + 顺序无关时顺便打开 interleave
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files) \
        .interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type=None),
            cycle_length=256,  # 同时打开的文件数
            num_parallel_calls=256)

    # 2. shuffle / map / batch / prefetch
    ds = ds.shuffle(buffer_size=shuffle_buffer,
                    reshuffle_each_iteration=True)

    ds = ds.map(parse_example_fn,
                num_parallel_calls=256)   # ← 关键并行`
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# 5) params（包含 all_column）
def build_params():
    all_column = build_model_columns_like_feature_process()
    # Keras 模型内部单独处理 userid，这里不放入 all_column
    if 'userid' in all_column:
        all_column.pop('userid')
    return {
        'all_column': all_column
    }


# 6) 训练
def train(tfrecord_files, epochs=1, batch_size=1024, lr=1e-4):
    params = build_params()
    model = build_mmoe_keras_model(params)


    #from tensorflow.keras import backend as K
    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #K.set_session(sess)  

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
                  loss=losses, loss_weights=loss_weights)#, metrics=metrics)

    ds = build_dataset(tfrecord_files, batch_size=batch_size)
    model.fit(ds, epochs=epochs)

    return model

from time import time
if __name__ == "__main__":
    files = [
        # 'hdfs://.../part-00000.tfrecord',
        "E:/work_space/reco-video-ovs-train/local_train/deeprank/tfrecord_data/part-00017-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord"
    ]


    files=tf.io.gfile.glob("hdfs://nameservice1/user/root/dingpan/add_uid_model/data/test_tfrecord/*")
    #files=tf.io.gfile.glob("hdfs://nameservice1/user/root/dingpan/add_uid_model/data/train_data/20251017_00/*")
    print("-----------------------files----------------------------------")
    print(files)
    files=[j for j in files if j.endswith(".tfrecord")]

    t1=time()
    model = train(files, epochs=1, batch_size=1024, lr=1e-4)
    t2=time()
    print(f"train time------------------------------------: {t2-t1}")

    print("begin save-------------------------------")
    
    # 创建保存目录
    import os
    import tensorflow as tf
    save_path = 'hdfs://nameservice1/user/root/dingpan/tf/mmoe_model.h5'
    #model.save(save_path)
    """
    # 使用 tf.saved_model.save 直接保存模型
    try:
        # 创建签名
        @tf.function
        def serving_fn(inputs):
            return model(inputs)
            
        # 保存模型
        tf.saved_model.save(
            model,
            save_path,
            signatures={'serving_default': serving_fn.get_concrete_function(
                {k: tf.TensorSpec(shape=(None,), dtype=v.dtype) for k, v in model.input.items()}
            )}
        )
        print(f"Model successfully saved in TF format at {save_path}")
    except Exception as e:
        print(f"Error saving model in TF format: {e}")
        # 备份方案：保存模型权重
        model.save_weights('mmoe_model_weights')
        print("Model weights saved as backup")

    """