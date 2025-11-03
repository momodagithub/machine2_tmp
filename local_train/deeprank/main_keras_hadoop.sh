

#!/bin/sh
set -x


export HADOOP_CONF_DIR=/etc/hadoop/conf1
export HADOOP_USER_NAME=hdfs


export LD_LIBRARY_PATH=/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/lib64:$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/dingpan/env/hdfs_so:/data/dingpan/env/jdk1.8.0_181/jre/lib/amd64/server
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)
#export CLASSPATH=/data/dingpan/box/core/target/hbox-core-1.9.6-SNAPSHOT.jar,/opt/cloudera/parcels/CDH-6.0.1-1.cdh6.0.1.p0.590678/jars/hadoop-common-3.0.0-cdh6.0.1.jar 
#conda activate spark_tf36
###tf 1.15.0
 python main_keras_hadoop.py
