

#!/bin/sh
set -x

HBOX_HOME=/data/dingpan/box/core/target/hbox-1.9.6-SNAPSHOT
export HADOOP_CONF_DIR=/etc/hadoop/conf1
export HADOOP_USER_NAME=hdfs

start_date=2025-10-28
end_date=2025-10-28
train_date=20251017
last_date=`date -d "${train_date} -1 day" +%Y-%m-%d`
last_model_path="hdfs://nameservice1/user/root/dingpan/add_uid_model/model/${last_date}/${hour}"
dt=${train_date}
hour="00"

last_model_path=""
output_model_dir="./add_uid_model/model/${dt}/${hour}"
output_model_dir="./add_uid_model/model/${dt}/${hour}"
summary_log_path="./add_uid_model/log/${dt}/${hour}"

train_global_step=6000
model_name=add_uid_model
##################################使用少的数据量
instance_inpath=/storage/dingpan/${dt}_${hour}

#export LD_LIBRARY_PATH=/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/lib64:$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/dingpan/env/hdfs_so:/data/dingpan/env/jdk1.8.0_181/jre/lib/amd64/server
#export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)
#export CLASSPATH=/data/dingpan/box/core/target/hbox-core-1.9.6-SNAPSHOT.jar,/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/jars/hadoop-common-3.0.0-cdh6.0.1.jar 
#conda activate spark_tf36
###tf 1.15.0
 python deeprank/main_train.py --model_name=$model_name  \
--start_date=${start_date} --end_date=${end_date} --last_model_path=${last_model_path} --hdfs_model_path=${output_model_dir}  \
--train_step=${train_global_step} --summary_log_path=${summary_log_path}  --data_path=${instance_inpath}
