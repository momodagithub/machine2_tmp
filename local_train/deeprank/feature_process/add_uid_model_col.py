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
        #if v=="userid":
        #    continue
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype=tf.int64 if f_name.lower() in int_sparse_feature else tf.string
            ),
            dimension=16
        )
        feature_columns[f_name.lower()] = f_embedding
    
    """
    userid_embedding=create_sharded_embedding_columns(
            fc.categorical_column_with_hash_bucket(
                'userid', hash_bucket_size=10000000, dtype=tf.string
            ),
            dimension=16, num_ps_shards=5
        )
    feature_columns['userid']=userid_embedding
    """

    ########################处理seq sparse特征#################################处理成了整型
    for k, v in varlen_columns.items():
        f_name=v.name
        f_embedding=fc.embedding_column(
            fc.categorical_column_with_hash_bucket(
                f_name, hash_bucket_size=v.vocabulary_size, dtype= tf.int64
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
