# coding:utf-8
import feature_process.common
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.python.ops import math_ops
from config.feature_column import *
def max_min_normalizer_fn(x_val, min_val, max_val):
    return (tf.cast(x_val, tf.float32) - min_val) / (max_val - min_val)


def quantile_normalizer_fn(x_val, boundaries):
    buckets = math_ops._bucketize(x_val, boundaries=boundaries)
    return tf.cast(buckets, tf.float32) / (len(boundaries))


def min_max_scaling(x):
    # 创建一个TensorFlow常量，用于比较和计算
    zero = tf.constant(0, dtype=x.dtype)
    divisor = tf.constant(222106, dtype=x.dtype)

    # 小于0则取0
    # 确保zero的形状与x相匹配，以便广播
    zero_filled = tf.fill(tf.shape(x), zero)
    x = tf.where(x < zero, zero_filled, x)

    # 先进行归一化
    x = x / divisor

    # 再取倒数
    res = 1 / (x + 1)

    return tf.cast(res, tf.float32)



import tensorflow as tf

# 假设 PS 节点数
NUM_PS = 5
EMB_DIM = 16
from tensorflow.python.feature_column import feature_column as fc_internal 
def partitioned_embedding_column(
        categorical_column,
        dimension,
        features,
        num_ps=NUM_PS,
        default_name='embedding'):
    """
    等价功能：fc.embedding_column，但变量被强制分区到所有 PS
    """
    # 1. 先拿到原始列名 & 真实 vocab_size
    vocab_size = categorical_column.num_buckets
    key = categorical_column.key

    # 2. 构造分区器：按“最小-最大”策略，每片最大 512 MB 或均匀分 num_ps 片
    partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps,        # 片数 = PS 台数
        axis=0,                       # 沿 vocab 轴切
        min_slice_size=64 * 1024 * 1024)  # 64 MB 最小片，可调

    # 3. 创建分区变量（会被 TF 自动平均放到不同 PS）
    with tf.variable_scope(default_name, partitioner=partitioner):
        embedding_weights = tf.get_variable(
            name=key + '_weights',
            shape=[vocab_size, dimension],
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            trainable=True)

    # 4. 手动 lookup，保持与 fc.embedding_column 相同输出 shape
    sparse_tensors = categorical_column._get_sparse_tensors(
        fc_internal._LazyBuilder(features), None)
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor  # None 或权重
    embeddings = tf.nn.embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner='mean')

    return embeddings
from tensorflow.python.ops import embedding_ops
def partitioned_embedding_column1(categorical_column,
                                 dimension=EMB_DIM,
                                 num_ps=NUM_PS,
                                 default_name='embedding'):
    """
    与 tf.feature_column.embedding_column 接口 100% 对齐的闭包实现。
    外层：只保存元信息；
    内层：在 Estimator 调用 input_layer / linear_model 时，由 TF 把 features
         注入，再执行分区变量 + lookup。
    """
    # ---------- 1. 校验 ----------
    if dimension is None or dimension < 1:
        raise ValueError('Invalid dimension {}.'.format(dimension))

    # ---------- 2. 构造内层闭包 ----------
    def _creator(weight_collections, scope):        # 签名与官方保持一致
        """内层函数：真正创建变量 & 返回 dense tensor。"""
        # 2-1. 分区器
        partitioner = tf.min_max_variable_partitioner(
            max_partitions=num_ps,
            axis=0,
            min_slice_size=64 << 20)   # 64 MB

        # 2-2. 创建分区变量（PS 会自动分片）
        with tf.variable_scope(default_name, partitioner=partitioner):
            embedding_weights = tf.get_variable(
                name=categorical_column.key + '_weights',
                shape=[categorical_column._num_buckets, dimension],
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True,
                collections=weight_collections)

        # 2-3. 官方同款：用 _LazyBuilder 把 features 转成 SparseTensor
        #      注意：这里不再传 features，而是由 TF 在 input_layer 阶段
        #      把 features 注入到 builder 里。
        builder = fc_internal._LazyBuilder(scope)   # scope 里自带 features
        sparse_tensors = categorical_column._get_sparse_tensors(builder, None)
        sparse_ids   = sparse_tensors.id_tensor
        sparse_weights = sparse_tensors.weight_tensor

        # 2-4. lookup 返回 dense tensor
        return embedding_ops.safe_embedding_lookup_sparse(
            embedding_weights,
            sparse_ids,
            sparse_weights,
            combiner='mean',
            name=default_name + '_lookup')
    
    # ---------- 3. 返回官方同款 _EmbeddingColumn 对象 ----------
    # 复用 TF 1.15 内部类，这样 input_layer/linear_model 都能识别
    return fc_internal._EmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner='mean',
        layer_creator=_creator,          # 把我们的闭包传进去
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True)

def partitioned_embedding_column2(categorical_column,
                                 dimension=16,
                                 num_ps=5,
                                 default_name='embedding'):
    """
    与 tf.feature_column.embedding_column 接口完全一致，
    但 embedding 变量会被分区到所有 PS。
    """
    if dimension is None or dimension < 1:
        raise ValueError('Invalid dimension {}.'.format(dimension))

    def _creator(weight_collections, scope):
        partitioner = tf.min_max_variable_partitioner(
            max_partitions=num_ps, axis=0,
            min_slice_size=64 << 20)

        with tf.variable_scope(scope or default_name, partitioner=partitioner):
            embedding_weights = tf.get_variable(
                name=categorical_column.key + '_weights',
                shape=[categorical_column._num_buckets, dimension],
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True,
                collections=weight_collections)

        # 让官方帮你 new _LazyBuilder，别再自己传 scope
        return categorical_column._get_dense_tensor(
            builder=None,
            weight_collections=weight_collections,
            trainable=True)

    # 返回官方同款 _EmbeddingColumn 对象
    from tensorflow.python.feature_column.feature_column import _EmbeddingColumn
    return _EmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner='mean',
        layer_creator=_creator,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True)
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn
from tensorflow.python.ops import variable_scope


def partitioned_embedding_column3(categorical_column,
                                 dimension=EMB_DIM,
                                 num_ps=NUM_PS,
                                 combiner='mean',
                                 initializer=None,
                                 ckpt_to_load_from=None,
                                 tensor_name_in_ckpt=None,
                                 max_norm=None,
                                 trainable=True):
    """
    与 tf.feature_column.embedding_column 接口 100% 相同，
    但 embedding 变量会被分区到所有 PS。
    """

    # ---- 1. 先拿到官方 embedding_column 对象 ----
    base_emb = tf.feature_column.embedding_column(
        categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable)

    # ---- 2. 偷梁换柱：把它的 layer_creator 包一层，只改 partitioner ----
    base_creator = base_emb.layer_creator          # 官方原来的 _creator

    def _partitioned_creator(weight_collections, scope):
        # 2-1 在 get_variable 之前临时注入 partitioner
        with variable_scope.variable_scope(
                scope or base_emb.name,
                partitioner=tf.min_max_variable_partitioner(
                    max_partitions=num_ps,
                    axis=0,
                    min_slice_size=64 << 20)):
            # 2-2 再调用官方 creator，此时 get_variable 会走上面的 partitioner
            return base_creator(weight_collections,
                                variable_scope.get_variable_scope())

    # ---- 3. 重新组装一个 _EmbeddingColumn，只换 layer_creator ----
    return _EmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        layer_creator=_partitioned_creator,   # 换成我们的
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable)

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops

NUM_PS = 5          # 实际 PS 数
EMB_DIM = 16        # 维度

def partitioned_embedding_column5(categorical_column,
                                 dimension=EMB_DIM,
                                 num_ps=NUM_PS,
                                 combiner='mean',
                                 initializer=None,
                                 ckpt_to_load_from=None,
                                 tensor_name_in_ckpt=None,
                                 max_norm=None,
                                 trainable=True):
    """
    与 tf.feature_column.embedding_column 接口完全一致，
    但 embedding 变量会被分区到所有 PS。
    """
    if dimension is None or dimension < 1:
        raise ValueError('Invalid dimension {}.'.format(dimension))

    # 1. 默认初始化器与官方一致
    if initializer is None:
        initializer = init_ops.truncated_normal_initializer(
            mean=0.0, stddev=1.0 / math.sqrt(dimension))

    # 2. 分区器
    partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps, axis=0,
        min_slice_size=64 << 20)   # 64 MB

    # 3. 官方同款 _creator 签名：(weight_collections, scope)
    def _creator(weight_collections, scope):
        with variable_scope.variable_scope(
                scope or categorical_column.key + '_embedding',
                partitioner=partitioner):
            embedding_weights = variable_scope.get_variable(
                name='embedding_weights',
                shape=[categorical_column._num_buckets, dimension],
                dtype=tf.float32,
                initializer=initializer,
                trainable=trainable,
                collections=weight_collections)
        return embedding_weights

    # 4. 直接构造官方 _EmbeddingColumn，只换 layer_creator
    return _EmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        layer_creator=_creator,               # 我们自己的
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable)


import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn
class ShardedEmbeddingColumn(_EmbeddingColumn):
    """支持将embedding矩阵分片到多个PS的自定义embedding column"""
    
    def __init__(self, 
                 categorical_column,
                 dimension, 
                 num_ps_shards=4,
                 initializer=None,
                 combiner='mean',
                 trainable=True,
                 name=None):
        """
        参数:
            categorical_column: 分类特征列
            dimension: embedding维度
            num_ps_shards: PS分片数量
            initializer: 初始化器
            combiner: 组合方式 ('mean', 'sqrtn', 'sum')
            trainable: 是否可训练
            name: 名称
        """
        super(ShardedEmbeddingColumn, self).__init__()
        self._categorical_column = categorical_column
        self._dimension = dimension
        self._num_ps_shards = num_ps_shards
        self._initializer = initializer or tf.truncated_normal_initializer(stddev=0.02)
        self._combiner = combiner
        self._trainable = trainable
        self._name = name or f"sharded_embedding_{categorical_column.name}"
        
    @property
    def name(self):
        return self._name
    
    @property
    def _var_scope_name(self):
        return self._name
    
    @property
    def _num_buckets(self):
        return self._categorical_column._num_buckets
    
    @property
    def _embedding_shape(self):
        return [self._num_buckets, self._dimension]
    
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        """
        获取embedding后的稠密张量
        """
        # 获取稀疏特征输入
        sparse_tensors = self._categorical_column._get_sparse_tensors(inputs)
        
        # 创建分片的embedding变量
        embedding_weights = partitioned_variables.create_partitioned_variables(
            shape=self._embedding_shape,
            slicing=[self._num_ps_shards, 1],  # 按行分片
            initializer=self._initializer,
            trainable=self._trainable,
            dtype=tf.float32,
            collections=weight_collections,
            name=self._name + "_weights"
        )
        
        # 使用embedding_lookup进行分片查找
        if isinstance(sparse_tensors, tuple):
            id_tensor = sparse_tensors.id_tensor
            weight_tensor = sparse_tensors.weight_tensor
        else:
            id_tensor = sparse_tensors
            weight_tensor = None
            
        # 分布式embedding查找
        embeddings = embedding_ops.safe_embedding_lookup_sparse(
            embedding_weights,
            id_tensor,
            sparse_weights=weight_tensor,
            combiner=self._combiner,
            partition_strategy='div'  # 使用除法策略进行分片
        )
        
        return embeddings
    
    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        """
        处理序列数据的embedding
        """
        sparse_tensors = self._categorical_column._get_sparse_tensors(inputs)
        
        embedding_weights = partitioned_variables.create_partitioned_variables(
            shape=self._embedding_shape,
            slicing=[self._num_ps_shards, 1],
            initializer=self._initializer,
            trainable=self._trainable,
            dtype=tf.float32,
            collections=weight_collections,
            name=self._name + "_weights"
        )
        
        if isinstance(sparse_tensors, tuple):
            id_tensor = sparse_tensors.id_tensor
            weight_tensor = sparse_tensors.weight_tensor
        else:
            id_tensor = sparse_tensors
            weight_tensor = None
            
        # 序列embedding查找
        embeddings = embedding_ops.embedding_lookup_sparse(
            embedding_weights,
            id_tensor,
            sparse_weights=weight_tensor,
            combiner=self._combiner,
            partition_strategy='div'
        )
        
        return tf.nn.embedding_lookup(embedding_weights, id_tensor)

def create_sharded_embedding_columns(categorical_columns, dimension, num_ps_shards=4):
    """
    批量创建分片embedding columns的便捷函数
    """
    return [
        ShardedEmbeddingColumn(
            cat_col,
            dimension,
            num_ps_shards,
            name=f"sharded_embedding_{cat_col.name}"
        ) for cat_col in categorical_columns
    ]


def build_model_columns(FLAGS):
    """Builds a set of wide and deep feature columns."""
    feature_columns = {}
    """
    userid_embedding = fc.embedding_column(
        fc.categorical_column_with_hash_bucket(
            'userid', hash_bucket_size=50000000, dtype=tf.string
        ),
        16
    )
    feature_columns['userid'] = userid_embedding
    """
    #################################处理dense 特征，dense特征已经进行了离散化#################################
    ###dense 特征在生成tfrecord时已经进行了离散化，所以这里直接使用identity
    for k, v in dense_columns.items():
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_identity(
                f_name, num_buckets=v.vocabulary_size
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding



    ###区分整数类型和string类型
    ########################处理sparse 特征#################################
    for k, v in sparse_columns.items():
        if v=="userid":
            continue
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.int64 if f_name.lower() in int_sparse_feature else tf.string
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding
    

    userid_embedding=create_sharded_embedding_columns(
            fc.categorical_column_with_hash_bucket(
                'userid', hash_bucket_size=10000000, dtype=tf.string
            ),
            dimension=16, num_ps_shards=5
        )
    feature_columns['userid']=userid_embedding


    ########################处理seq sparse特征#################################处理成了整型
    for k, v in varlen_columns.items():
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.int64
            ),
            dimension=16,combiner='mean'
        )
        feature_columns[f_name.lower()] = f_embedding
    ########################处理rank 特征#################################
    for k, v in rank_columns.items():
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.string
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding
    ########################处理position 特征#################################
    for v in position_columns:
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.int64
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding


    #########逻辑暂时不写在这里#################################
    return feature_columns


def build_label_columns():
    label_names = ["ctr_label", "cvr_label", "vip_label", "unlock_label", "finish_view_label"]
    label_ctr = fc.numeric_column('ctr_label', dtype=tf.int64)
    label_cvr = fc.numeric_column('cvr_label', dtype=tf.int64)
    label_vip = fc.numeric_column('vip_label', dtype=tf.int64)
    label_unlock=fc.numeric_column('unlock_label', dtype=tf.int64)
    label_finish=fc.numeric_column('finish_view_label', dtype=tf.int64)
    label_columns={}
    label_columns['ctr_label']=label_ctr
    label_columns['cvr_label']=label_cvr
    label_columns['vip_label']=label_vip
    label_columns['unlock_label']=label_unlock
    label_columns['finish_view_label']=label_finish
    return label_columns


def build_loss_weight_columns(FLAGS):
    loss_weight = fc.numeric_column(FLAGS.loss_weight_name, dtype=tf.float32)
    return loss_weight
