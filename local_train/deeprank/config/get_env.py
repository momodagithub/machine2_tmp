import argparse
import numpy as np
import tensorflow as tf



import os
import json
import time
cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
task_index = int(os.environ["TF_INDEX"])
task_type = os.environ["TF_ROLE"]

tf_config = dict()
worker_num = len(cluster["worker"])

flag_ps=False
if task_type == "ps":
    flag_ps=True
    tf_config["task"] = {"index":task_index, "type":task_type}
elif task_type == "worker":
    if task_index == 0:
        tf_config["task"] = {"index":0, "type":"chief"}
    else:
        #time.sleep(300)
        tf_config["task"] = {"index":task_index-1, "type":task_type}
elif task_type == "evaluator":
    tf_config["task"] = {"index":task_index, "type":task_type}

if worker_num == 1:
    cluster["chief"] = cluster["worker"]
    del cluster["worker"]
else:
    cluster["chief"] = [cluster["worker"][0]]
    del cluster["worker"][0]

tf_config["cluster"] = cluster
os.environ["TF_CONFIG"] = json.dumps(tf_config)
print(json.loads(os.environ["TF_CONFIG"]))

#######文件输入
import os
import json
if task_type!="ps":
    if "INPUT_FILE_LIST" in os.environ:
        inputfile = json.loads(os.environ["INPUT_FILE_LIST"])
        data_file = inputfile["data"]
    else:
        with open("inputFileList.txt") as f:
            fileStr = f.readline()
            data_file = json.loads(fileStr)
else:
    data_file=[]
#print("--------------------------")

