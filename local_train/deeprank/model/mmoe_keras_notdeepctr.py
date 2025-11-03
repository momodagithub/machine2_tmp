import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Softmax
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
# removed: from deepctr.layers.core import DNN
from tensorflow import feature_column as fc
from config.feature_column import sparse_columns, dense_columns, rank_columns, varlen_columns, position_columns

# 以下为新增：用 Keras 实现的 DNN 层，替换 deepctr 的 DNN
class DNN(layers.Layer):
    def __init__(self, hidden_units=(64,), activation='relu', dropout_rate=0.0, use_bn=False, kernel_initializer='glorot_uniform', name=None):
        super().__init__(name=name)
        if isinstance(hidden_units, int):
            hidden_units = (hidden_units,)
        self.hidden_units = tuple(hidden_units)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        self._dense_layers = []
        self._bn_layers = []
        self._dropout_layers = []

        for i, unit in enumerate(self.hidden_units):
            self._dense_layers.append(Dense(unit, activation=self.activation, kernel_initializer=self.kernel_initializer, name=(None if name is None else f"{name}_dense_{i}")))
            if self.use_bn:
                self._bn_layers.append(layers.BatchNormalization(name=(None if name is None else f"{name}_bn_{i}")))
            else:
                self._bn_layers.append(None)
            if self.dropout_rate and self.dropout_rate > 0.0:
                self._dropout_layers.append(layers.Dropout(self.dropout_rate, name=(None if name is None else f"{name}_dropout_{i}")))
            else:
                self._dropout_layers.append(None)

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for dense, bn, drop in zip(self._dense_layers, self._bn_layers, self._dropout_layers):
            x = dense(x)
            if bn is not None:
                x = bn(x, training=training)
            if drop is not None:
                x = drop(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_bn": self.use_bn,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ...existing code...
def _get_attention(query_input, key_input, att_emb_size=128):
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


class Expert(layers.Layer):
    def __init__(self, expert_hidden_units, output_units, name="expert"):
        super().__init__(name=name)
        self.expert_hidden_units = expert_hidden_units
        self.output_units = output_units
        self.expert_layers = [Dense(units, activation='relu') for units in expert_hidden_units]
        self.output_layer = Dense(output_units, activation='relu')

    def call(self, inputs, **kwargs):
        x = inputs
        for layer_ in self.expert_layers:
            x = layer_(x)
        return self.output_layer(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "expert_hidden_units": self.expert_hidden_units,
            "output_units": self.output_units
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GatingNetwork(layers.Layer):
    def __init__(self, num_experts, name="gating"):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.gate = Dense(num_experts, activation='softmax')

    def call(self, inputs, **kwargs):
        return self.gate(inputs)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiExpertLayer(layers.Layer):
    def __init__(self, num_experts, expert_hidden_units, expert_output_units, name="multi_expert"):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.expert_hidden_units = expert_hidden_units
        self.expert_output_units = expert_output_units
        self.experts = [Expert(expert_hidden_units, expert_output_units, name=f"expert_{i}") for i in range(num_experts)]

    def call(self, inputs, **kwargs):
        outputs = [expert(inputs) for expert in self.experts]
        return tf.stack(outputs, axis=1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts,
            "expert_hidden_units": self.expert_hidden_units,
            "expert_output_units": self.expert_output_units
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_mmoe_keras_model(params):
    # columns
    feat_columns = {**sparse_columns, **dense_columns, **rank_columns, **varlen_columns}
    sparse_feature_columns = [c for c in feat_columns.values() if isinstance(c, SparseFeat)]
    varlen_sparse_feature_columns = [c for c in feat_columns.values() if isinstance(c, VarLenSparseFeat)]
    gate_columns = ["language", "book_language", "vip_type", "shelf_days", "register_min", "click_pv_7days", "prov", "page", "ad_gap_min", "uVipLabel", "activeDay", "activeDays7d"]
    query_columns = ["book_id", "gender", "gap_day"]
    seq_columns = ['userWatchBookListLatestBookId', 'userWatchBookListLatestGapDay', 'userWatchBookListLatestGender',
                   'userFollowBookListLatestBookId', 'userFollowBookListLatestGapDay', 'userFollowBookListLatestGender',
                   'userAttributeBookListLatestBookId', 'userAttributeBookListLatestGapDay', 'userAttributeBookListLatestGender',
                   'userClickBookListLatestBookId', 'userClickBookListLatestGapDay', 'userClickBookListLatestGender',
                   'userFullWatchBookListLatestBookId', 'userFullWatchBookListLatestGapDay', 'userFullWatchBookListLatestGender',
                   'userBuyBookListLatestBookId', 'userBuyBookListLatestGapDay', 'userBuyBookListLatestGender',
                   'userOnlyUnFollowBookListLatestBookId', 'userOnlyUnFollowBookListLatestGapDay', 'userOnlyUnFollowBookListLatestGender',
                   'userFreeFullWatchBookListLatestBookId', 'userFreeFullWatchBookListLatestGapDay', 'userFreeFullWatchBookListLatestGender',
                   'userLastWatchBookListBookId', 'userLastWatchBookListGapDay', 'userLastWatchBookListGender',
                   'userWatchBookListGt5EpdBookId', 'userWatchBookListGt5EpdGapDay', 'userWatchBookListGt5EpdGender',
                   'userWatchBookListGt10EpdBookId', 'userWatchBookListGt10EpdGapDay', 'userWatchBookListGt10EpdGender',
                   'userFullWatchBookListLatestCheckedBookId', 'userFullWatchBookListLatestCheckedGapDay', 'userFullWatchBookListLatestCheckedGender']

    # Keras Inputs
    inputs = {}
    # sparse features: 区分 int 和 string 类型
    for c in sparse_columns.values():
        if c.name=="userid":
            continue
        if c.name.lower() in ['week_day', 'hour', 'month_day', "pappversion", "userlatestvideolistlabels"]:
            inputs[c.name] = tf.keras.Input(shape=(), dtype=tf.int32, name=c.name)
        else:
            inputs[c.name] = tf.keras.Input(shape=(), dtype=tf.string, name=c.name)
    
    # dense features: 全部是 int 类型
    for c in dense_columns.values():
        inputs[c.name] = tf.keras.Input(shape=(), dtype=tf.int32, name=c.name)
    
    # rank features: 全部是 string 类型
    for c in rank_columns.values():
        inputs[c.name] = tf.keras.Input(shape=(), dtype=tf.string, name=c.name)
    
    # varlen int features as sparse inputs
    for c in varlen_sparse_feature_columns:
        inputs[c.name] = tf.keras.Input(shape=(c.maxlen,), dtype=tf.int32, sparse=True, name=c.name)
    
    # raw string userid
    inputs['userid'] = tf.keras.Input(shape=(), dtype=tf.string, name='userid')

    # fc.input_layer can work with Keras Tensors
    feature_columns_tf = params['all_column']

    emb_cache = {}

    # userid via Keras Embedding with online hashing
    num_buckets = 10000000
    userid_ids = tf.strings.to_hash_bucket_fast(inputs['userid'], num_buckets)
    userid_ids = tf.cast(userid_ids, tf.int32)
    userid_embed = layers.Embedding(input_dim=num_buckets, output_dim=16,
                                    embeddings_initializer=tf.keras.initializers.glorot_normal(),
                                    name='userid_feature_embeddings')(userid_ids)
    userid_embed = tf.reshape(userid_embed, (-1, 16))
    emb_cache['userid'] = userid_embed
    # include into sparse features
    #sparse_embedding_list0.append(userid_embed)
    # 替换后：
    from tensorflow.keras.layers import DenseFeatures
    def get_cached_emb(name):
        key = name.lower()
        if key in emb_cache:
            return emb_cache[key]
        #this_emb = fc.input_layer(inputs, [feature_columns_tf[key]])
        this_emb = DenseFeatures(feature_columns_tf[key])(inputs)
        emb_cache[key] = this_emb
        return this_emb

    # shared embeddings
    sparse_embedding_list0 = [get_cached_emb(f.name) for f in sparse_feature_columns]
    gate_embedding_list0 = [get_cached_emb(n) for n in gate_columns if n in feature_columns_tf]

    # query/keys
    query_emb_base = [get_cached_emb(n) for n in query_columns if n in feature_columns_tf]
    repeats = int(len(seq_columns) / max(len(query_columns), 1))
    query_emb_list0 = query_emb_base * repeats
    keys_emb_list = [get_cached_emb(n) for n in seq_columns if n in feature_columns_tf]

    # varlen non-seq features
    non_seq_varlen = []
    for c in varlen_sparse_feature_columns:
        if c.name not in seq_columns:
            non_seq_varlen.append(c)
    sequence_embed_list = [get_cached_emb(f.name) for f in non_seq_varlen]

    

    # gating and attention
    query_input = tf.concat(query_emb_list0, axis=-1) if len(query_emb_list0) > 0 else tf.zeros_like(userid_embed)
    key_input = tf.concat(keys_emb_list, axis=-1) if len(keys_emb_list) > 0 else tf.zeros_like(tf.expand_dims(userid_embed, 1))
    attention_emb = _get_attention(query_input, key_input)

    inputs_concat = tf.concat(sparse_embedding_list0 + sequence_embed_list, axis=1) if (sparse_embedding_list0 or sequence_embed_list) else userid_embed
    gate_inputs = tf.concat(gate_embedding_list0, axis=1) if len(gate_embedding_list0) > 0 else userid_embed

    feat_num = len(sparse_embedding_list0 + sequence_embed_list)
    inputs_reshape = tf.reshape(inputs_concat, (-1, feat_num, 16))

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
    ctr_expert_gate = GatingNetwork(4, name="gate_ctr")(ctr_input)
    experts_for_ctr = tf.concat([shared_out, ctr_out], axis=1)
    ctr_emb = tf.reduce_sum(tf.expand_dims(ctr_expert_gate, -1) * experts_for_ctr, axis=1)

    fvtr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="fvtr_experts")(finish_input)
    fvtr_expert_gate = GatingNetwork(4, name="gate_fvtr")(finish_input)
    experts_for_fvtr = tf.concat([shared_out, fvtr_out], axis=1)
    finish_emb = tf.reduce_sum(tf.expand_dims(fvtr_expert_gate, -1) * experts_for_fvtr, axis=1)

    utr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="utr_experts")(unlock_input)
    utr_expert_gate = GatingNetwork(4, name="gate_unlock")(unlock_input)
    experts_for_unlock = tf.concat([shared_out, utr_out], axis=1)
    unlock_emb = tf.reduce_sum(tf.expand_dims(utr_expert_gate, -1) * experts_for_unlock, axis=1)

    cvr_out = MultiExpertLayer(num_experts=1, expert_hidden_units=[512, 256, 128], expert_output_units=64, name="cvr_experts")(cvr_input)
    cvr_expert_gate = GatingNetwork(4, name="gate_cvr")(cvr_input)
    experts_for_cvr = tf.concat([shared_out, cvr_out], axis=1)
    cvr_emb = tf.reduce_sum(tf.expand_dims(cvr_expert_gate, -1) * experts_for_cvr, axis=1)

    ctr = layers.Dense(1, activation="sigmoid", use_bias=False, name='ctr_out')(ctr_emb)
    fvtr = layers.Dense(1, activation="sigmoid", use_bias=False, name='finish_out')(finish_emb)
    unlock = layers.Dense(1, activation="sigmoid", use_bias=False, name='unlock_out')(unlock_emb)
    cvr = layers.Dense(1, activation="sigmoid", use_bias=False, name='cvr_out')(cvr_emb)

    model = tf.keras.Model(inputs=inputs, outputs={"ctr": ctr, "fvtr": fvtr, "unlock": unlock, "cvr": cvr}, name="mmoe_keras")
    return model