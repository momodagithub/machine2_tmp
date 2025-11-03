# coding:utf-8
import tensorflow as tf
from config.feature_column import *

def getFLAG():
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 512, "Training batch size")
    flags.DEFINE_float("learning_rate", 1e-1, "learning rate")
    flags.DEFINE_float("l2_reg", 1e-4, "weight decay")
    flags.DEFINE_integer("shuffle_buffer_size", 100000, "dataset shuffle buffer size")
    flags.DEFINE_integer("max_steps", 200000000, "max training steps")
    flags.DEFINE_integer("eval_steps", 1, "evaluation steps")
    flags.DEFINE_integer("num_epochs", 1, "the number of training dataset epoch")
    flags.DEFINE_integer("save_checkpoint_steps", 500, "save checkpoint steps")
    flags.DEFINE_string("dt_start", "2024-06-14", "dt start")
    flags.DEFINE_string("dt_end", "2024-06-18", "dt end")
    flags.DEFINE_string(
        "model_dir",
        "hdfs://datalake/user/hdp-omi-tech/male_like_32_v3/model/{}",
        "Base directory for the model."
    )
    flags.DEFINE_string(
        "export_dir",
        "hdfs://datalake/user/hdp-omi-tech/male_like_32_v3/export/{}",
        "Base directory for the exported model."
    )
    flags.DEFINE_string(
        "train_data",
        "hdfs://datalake/user/hdp-omi-tech/male_like_v1_32_clip_ovis/data/default/{}/train_data/*",
        "Path to the training data."
    )
    flags.DEFINE_integer("user_id_emb_dim", 16, "user id emb dim")
    flags.DEFINE_integer("user_id_hash_size", 40000000, "user id num")


    dense_feature = [v[0] for v in dense_discretized_slot.values()]
    sparse_feature = [v.name for v in sparse_columns.values()]
    rank_feature = [v.name for v in rank_columns.values()]
    varlen_feature = [v.name for v in varlen_columns.values()]
    flags.DEFINE_list("dense_feature", dense_feature, "dense select")
    flags.DEFINE_list("sparse_feature", sparse_feature, "sparse select")
    flags.DEFINE_list("rank_feature", rank_feature, "rank select")
    flags.DEFINE_list("varlen_feature", varlen_feature, "varlen select")



    FLAGS = flags.FLAGS

    return FLAGS
