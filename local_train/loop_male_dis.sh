set -x
source /etc/profile

start_date=2025-02-10
end_date=2025-06-04

# 将日期格式转换为时间戳
start_ts=$(date -d "$start_date" +%s)
end_ts=$(date -d "$end_date" +%s)
# 一天的秒数
one_day=$((60 * 60 * 24))
instance_inpath=""

# 循环从开始日期到结束日期
current_ts=$start_ts
#while [  "$current_ds" -le "$end_ts" ] ; do







while [ "$current_ts" -le "$end_ts" ]; do
    # 将时间戳转换回日期格式
        start_date=$(date -d "@$current_ts" +%Y-%m-%d)
   this_date=${start_date}
      # 设置要检查的 Hadoop 输出目录
   HADOOP_OUTPUT_DIR=hdfs://datalake/user/hdp-omi-tech/male_like_v1_32_clip_ovis/data/default/${start_date}/train_data

   # 设置检查间隔时间（秒）
   CHECK_INTERVAL=10

    # 循环检测
  while true; do
    # 检查 _SUCCESS 文件是否存在
    if hadoop fs -test -e "$HADOOP_OUTPUT_DIR/_SUCCESS"; then
        echo "_SUCCESS 文件已找到，Hadoop 作业完成！"
        break
    else
        echo "_SUCCESS 文件未找到，等待 $CHECK_INTERVAL 秒后重试..."
    fi

    # 等待指定的时间
    sleep $CHECK_INTERVAL
 done




    cd train_num

   ############# sh num.sh ${this_date} ${this_date}
    ##判断数量
    cd ..

 
    SUCCESS_FLAG=hdfs://datalake/user/hdp-omi-tech/dingpan/learning/male_like_3/${start_date}/_SUCCESS  
    ###获取训练数据量
      train_num=`head -1 train_num/num_${start_date}.txt | grep -o '[0-9]\+' | head -1`     
      train_step=$((train_num / 256))
      # 主循环
  while :; do
    # 检测HDFS上的成功标志
    if hdfs dfs -test -e "$SUCCESS_FLAG"; then
        echo "[INFO] 检测到成功标志，任务已成功完成！"

        ##判断checkpoint
         num1=`hdfs dfs -ls hdfs://datalake/user/hdp-omi-tech/dingpan/learning/male_like_3/${start_date}  | grep "model.ckpt-" | awk -F'ckpt-' '{print $2}' | awk -F'.' '{print $1}' | sort -n | tail -1`
        ###上一天的
        last_date=$(date -d "$start_date-1days" +%Y-%m-%d)
         num2=`hdfs dfs -ls hdfs://datalake/user/hdp-omi-tech/dingpan/learning/male_like_3/${last_date}  | grep "model.ckpt-" | awk -F'ckpt-' '{print $2}' | awk -F'.' '{print $1}' | sort -n | tail -1`
        if   [ $num1 -gt $num2 ]; then 
           break
        fi
    else
        sh run_train.sh ${start_date} ${train_step}
    fi
   done


      current_ts=$((current_ts + one_day))
done


