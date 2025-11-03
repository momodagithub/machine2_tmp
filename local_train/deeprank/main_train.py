#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import subprocess
"""
def setup_hadoop_env():
    try:
        result = subprocess.run(['hadoop', 'classpath'], capture_output=True, text=True)
        if result.returncode == 0:
            hadoop_classpath = result.stdout.strip()
            os.environ['CLASSPATH'] = hadoop_classpath
            os.environ['HADOOP_CLASSPATH'] = hadoop_classpath
            return True
    except Exception as e:
        print(f"Hadoop环境配置失败")
        return False

        
setup_hadoop_env()
"""
from data_input.common import train_input_fn, eval_input_fn
import argparse
import json
import datetime, time

###################################new import
#from config.get_env import *
from config.load_model import  RestoreCheckpointHook


from feature_process.add_uid_model_col import build_model_columns as build_add_uid_columns_col
from feature_process.add_uid_model_col import build_label_columns as build_add_uid_label_col
from feature_process.add_uid_model_col import build_loss_weight_columns as build_add_uid_loss_weight_col


from config.add_uid_model import getFLAG as add_uid_model_flag
#################from model.mmoe import model_fn as mmoe_model
#from model.mmoe_partion import model_fn as mmoe_model
from model.mmoe1 import model_fn as mmoe_model



from data_input.common  import train_input_fn #as common_train_input_fn
from data_input.common import eval_input_fn


def parse_args():
    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('--model_name', default='add_uid_model', type=str, help='模型名称')
    parser.add_argument('--mode', default='train', type=str, help='模式')
    parser.add_argument('--start_date', default='default', type=str, help='手动指定训练数据日期,如果传了则使用这个')
    parser.add_argument('--end_date', default='default', type=str, help='手动指定训练数据日期,如果传了则使用这个')
    #############################################################
    parser.add_argument('--hdfs_model_path', default='', type=str, help='')
    parser.add_argument('--train_step', default='1000', type=int, help='')
    parser.add_argument('--last_model_path', default='', type=str, help='')
    parser.add_argument('--summary_log_path', default='', type=str, help='')
    # Flags for defining the parameter of data path
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="The path for data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model",
        help="The save path for model"
    )

    # Flags for defining the parameter of train
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate of the train"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="the max_steps of the train"
    )

    args = parser.parse_args()
    return args

#####train_data_file=data_file
args1 = parse_args()
train_data_file=tf.io.gfile.glob(args1.data_path+"/*")
#train_data_file=['./test_tfrecord/part-00000-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00001-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00002-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00003-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00004-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00005-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00006-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00007-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00008-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00009-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00010-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00011-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00012-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00013-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00014-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00015-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00016-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00017-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00018-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord', './test_tfrecord/part-00019-cf958cef-67d9-4933-8c0a-1fe15a47a268-c000.tfrecord']

def train_eval_model(FLAGS, feature_columns, label, model,args):
    #fea_columns = feature_columns + [label]
    fea_columns = feature_columns + label
    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)

    # profilerHook = tf.estimator.ProfilerHook(save_steps=100, output_dir=FLAGS.model_dir)
    global train_data_file
    if args.last_model_path!='':
        hook=RestoreCheckpointHook(init_global_step=False,  restore_checkpoint_dir=args.last_model_path)
        train_hooks=[hook]
    else:
        train_hooks=[]

    print("------------------------------------------ok1")

    base_step = 0
    if tf.train.latest_checkpoint(args.last_model_path) is not None:
        base_step = int(tf.train.latest_checkpoint(args.last_model_path).split("-")[-1])
        #global_steps_per_epoch = np.int64(self.train_config.instance_num // self.train_config.batch_size)
        max_steps = base_step + args.train_step# global_steps_per_epoch * self.train_config.max_epochs 
    else:
        max_steps=args.train_step
    print("------------------------------------------ok2")

    ###if task_type=='worker' and task_index==0  and args.last_model_path!='':
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(
            filenames=train_data_file,#FLAGS.train_data.format(DATE),
            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            feature_spec=feature_spec,
            #hooks=train_hooks,
        ),

        max_steps=max_steps,
        # hooks=[profilerHook]
        hooks=train_hooks,
    )

    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(
            filenames=[train_data_file[0]],
            batch_size=FLAGS.batch_size,
            feature_spec=feature_spec
        ),
        steps=FLAGS.eval_steps
    )

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

def train(DATE, FLAGS, model, feature_columns, label):
    fea_columns = feature_columns + [label]
    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)

    model.train(
        input_fn=lambda: train_input_fn(
            filenames=FLAGS.train_data.format(DATE),
            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            feature_spec=feature_spec,
        ),
        steps=FLAGS.max_steps,
    )

def export_model_example(feature_columns, model, export_dir):
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=feature_columns)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(export_dir, serving_input_fn)

def export_model_raw(feature_columns, model, export_dir):
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    feature_map = {}
    for key, feature in feature_spec.items():
        if isinstance(feature, tf.io.VarLenFeature):  # 可变长度
            feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[1], name=key)
        elif isinstance(feature, tf.io.FixedLenFeature):  # 固定长度
            feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[None, feature.shape[0]], name=key)
    serving_input_recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
    model.export_savedmodel(export_dir, serving_input_recevier_fn)

def date_list_creater(start, end):
    result=[start]
    tmp = start
    while tmp < end:
        tmp = (datetime.datetime.strptime(tmp, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        result.append(tmp)
    return result


from config.feature_column import *
def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    label_names = ["ctr_label", "cvr_label", "vip_label", "unlock_label", "finish_view_label"]
    # feature_columns = sparse_columns + dense_columns + rank_columns + varlen_columns
    feature_columns_dic = {**sparse_columns, **dense_columns, **rank_columns, **varlen_columns}
    feature_columns = [feature_columns_dic[key] for key in sorted(feature_columns_dic)]
    feature_names = [feat.name for feat in feature_columns + position_columns]

    cname2fname = {}
    for _fname in feature_names:
        if _fname.lower() != _fname:
            cname2fname[_fname.lower()] = _fname

    feat_sig = {}
    BATCH_SIZE=1024
    for feat in feature_columns+position_columns:
        feat_sig[feat.name] = tf.TensorSpec(shape=[BATCH_SIZE,feat.maxlen], dtype=tf.int32, name=feat.name) if isinstance(feat,
                                                                                                               VarLenSparseFeat) else tf.TensorSpec(
            shape=[BATCH_SIZE,], dtype=tf.int32, name=feat.name)
    label_sig = {}
    for label in label_names:
        label_sig[label] = tf.TensorSpec(shape=[BATCH_SIZE,], dtype=tf.int32, name=label)
    #run_config = tf.estimator.RunConfig().replace(
    #    model_dir=model_path, log_step_count_steps=200, save_summary_steps=1000, save_checkpoints_steps=30000,
    #    keep_checkpoint_max=10)
    history_columns = ["book_id", "gender", "gap_day"]
    history_seq_columns = ['userWatchBookListLatestBookId', 'userWatchBookListLatestGapDay', 'userWatchBookListLatestGender'
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
    gate_input_column_names = ["language", "book_language", "vip_type", "shelf_days", "register_min", "click_pv_7days", "prov", "page", "ad_gap_min", "uVipLabel", "activeDay", "activeDays7d"]
    gate_input_columns = []
    for feat in feature_columns:
        if feat.name in gate_input_column_names:
            gate_input_columns.append(feat)



    args = parse_args()

    if args.model_name == 'add_uid_model':
        FLAGS1 = add_uid_model_flag()
        model_fn = mmoe_model
        feature_columns = build_add_uid_columns_col(FLAGS1)
        label_columns = build_add_uid_label_col()
        #######columns 合并dict
        all_columns={}
        for f in feature_columns:
            all_columns[f]=feature_columns[f]
        
        for f in label_columns:
            all_columns[f]=label_columns[f]
    else:
        print("invalid model_name ", args.model_name)
        return

    #start_date = FLAGS.dt_start
    #end_date = FLAGS.dt_end
    if args.start_date != 'default' and args.end_date != 'default':
        start_date = args.start_date
        end_date = args.end_date
        print("manually specify start_date end_date", start_date, end_date)

    tf.random.set_random_seed(42)
  
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,#FLAGS1.save_checkpoint_steps,
        keep_checkpoint_max=2,
        # session_config=session_config,
    )

    ######配置参数
    params = {
        "learning_rate": FLAGS1.learning_rate,
        "l2_reg": FLAGS1.l2_reg,
        "all_column":all_columns,
        "feature_columns": feature_columns, 
        "model_dir": args.summary_log_path, 
        "pos_columns": position_columns, 
        "query_columns": history_columns, 
        "seq_columns": history_seq_columns, 
        "gate_input_columns": gate_input_columns,
    }

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.hdfs_model_path,
        config=run_config,
        params=params
    )
    
    
    feature_columns_list =  [i for i in feature_columns.values()] 
    label_columns_list=[i for i in label_columns.values()] 
    train_eval_model( FLAGS1, feature_columns_list, label_columns_list,model,args)
    # 导出模型
    #if  task_type=="worker" and task_index==0:
    #    export_path =args.hdfs_model_path+"/serving_pb"# FLAGS1.export_dir
    #    export_model_example(feature_columns, model, export_path) 

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
