#coding:utf-8
import tensorflow as tf
from tensorflow import feature_column as fc
from lazyAdam import LazyAdamOptimizer

def model_fn(features, labels, mode, params):
    # 特征输入
    user_id = fc.input_layer(
        features, [params["feature_columns"]["actor_id"], params["feature_columns"]["receiver_id"]]
    )

    bias_feature_column_list = [
        params["feature_columns"][key]
        for key in params["FLAGS"].feature_select
        if key not in ['actor_id', 'receiver_id']
    ]
    bias = fc.input_layer(features, bias_feature_column_list)
    # print(bias.shape())

    actor_id_emb = user_id[:, :params["FLAGS"].user_id_emb_dim-1]
    actor_id_bias = tf.expand_dims(user_id[:, params["FLAGS"].user_id_emb_dim-1], axis=-1)

    receiver_id_emb = user_id[:, params["FLAGS"].user_id_emb_dim:2*params["FLAGS"].user_id_emb_dim-1]
    receiver_id_bias = tf.expand_dims(user_id[:, 2*params["FLAGS"].user_id_emb_dim-1], axis=-1)

    #内积
    dot = tf.reduce_sum(tf.multiply(actor_id_emb, receiver_id_emb), axis=1, keepdims=True)

    res = tf.concat([actor_id_bias, receiver_id_bias, dot, bias], axis=1)

    res = tf.reduce_sum(res, axis=1, keepdims=True)

    # 最后输出
    pred = tf.nn.sigmoid(res)

    if mode == tf.estimator.ModeKeys.PREDICT:
        outputDict = {}
        outputDict["score"] = pred
        outputDict["actor_id_emb"] = actor_id_emb
        outputDict["receiver_id_emb"] = receiver_id_emb
        outputDict["actor_id_bias"] = actor_id_bias
        outputDict["receiver_id_bias"] = receiver_id_bias
        outputDict["bias"] = bias
        return tf.estimator.EstimatorSpec(
            mode, predictions=outputDict
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.log_loss(labels, pred)
        auc = tf.metrics.auc(labels=labels, predictions=pred)

        # tf.summary.scalar("evalLoss", loss)
        # tf.summary.scalar("auc", auc)
        metrics = {"auc": auc}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        #交叉熵损失函数
        loss = tf.losses.log_loss(labels, pred)

        global_step = tf.train.get_global_step()
        optimizer = LazyAdamOptimizer(learning_rate=params["FLAGS"].learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)