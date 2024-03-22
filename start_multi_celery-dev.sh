#!/usr/bin/env bash

# 数组用于存储PID
pids=()

# 启动6个实例并记录PID
for i in {1..3}; do
  celery -A AudioProcessCeleryWorker worker -l WARNING -c 1 --pool=solo -Q GPTSoVits &
  pids+=($!)
done

# 定义函数用于重启实例
restart_instances() {
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null
    if [ $? -ne 0 ]; then
      echo "Restarting instance with PID: $pid"
      celery -A AudioProcessCeleryWorker worker -l WARNING -c 1 --pool=solo -Q GPTSoVits  &
      pids=("${pids[@]/$pid}")
      pids+=($!)
    fi
  done
}

# 定义函数用于强制终止实例
force_kill_instances() {
  echo "Force killing instances..."
  for pid in "${pids[@]}"; do
    kill -9 "$pid"
  done
  exit 0
}

# 监听Ctrl+C信号，触发强制终止实例
trap force_kill_instances SIGINT

# 持续检查并重启实例
while true; do
  restart_instances
  sleep 5
done