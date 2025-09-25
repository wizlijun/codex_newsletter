#!/bin/bash

# 周期运行器：每隔固定时间执行一次任务，并在控制台显示倒计时和详细日志。
# 每轮会先运行 getmail.py，随后立即运行 readmail.py。
#
# 用法示例：
#   - 直接运行（默认 5 分钟，默认使用 newsletter.yml）：
#       ./runloop.sh
#   - 指定间隔为 10 分钟：
#       ./runloop.sh 600
#   - 传递参数给 getmail.py：
#       ./runloop.sh 300 --config newsletter.yml -v

# 默认间隔时间（秒）
INTERVAL=${1:-300}
shift  # 移除第一个参数，剩下的都传给 getmail.py

# 如果没有传递额外参数，使用默认配置
if [ $# -eq 0 ]; then
    GETMAIL_ARGS="--config newsletter.yml"
else
    GETMAIL_ARGS="$@"
fi

# 工作目录切到脚本所在目录
cd "$(dirname "$0")"

# 检查 Python 解释器
if [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "错误: 找不到 Python 解释器"
    exit 1
fi

echo "================================================================================"
echo "[runloop] 配置："
echo "  - 间隔: $INTERVAL 秒"
echo "  - Python: $PYTHON"
echo "  - getmail 参数: $GETMAIL_ARGS"
echo "  - readmail 参数: (默认)"
echo "[runloop] 按 Ctrl+C 可随时退出。"
echo "================================================================================"

# 信号处理
trap 'echo -e "\n[runloop] 已收到中断信号，退出。"; exit 0' INT TERM

# 倒计时函数
countdown() {
    local seconds=$1
    for ((i=seconds; i>0; i--)); do
        local m=$((i / 60))
        local s=$((i % 60))
        printf "\r[runloop] 下次运行倒计时: %02d:%02d (按 Ctrl+C 退出)" $m $s
        sleep 1
    done
    printf "\r[runloop] 倒计时结束，开始下一轮运行...                    \n"
}

# 主循环
run_no=0
while true; do
    run_no=$((run_no + 1))
    
    echo "================================================================================"
    echo "[runloop] 第 $run_no 次运行开始 @ $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 运行 getmail.py
    echo "[runloop] 启动 getmail.py..."
    start_time=$(date +%s)
    if $PYTHON getmail.py $GETMAIL_ARGS; then
        rc_getmail=0
    else
        rc_getmail=$?
    fi
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "[runloop] getmail.py 结束，退出码 $rc_getmail，耗时 ${duration}s"
    
    # 运行 readmail.py
    echo "[runloop] 启动 readmail.py..."
    start_time=$(date +%s)
    if $PYTHON readmail.py; then
        rc_readmail=0
    else
        rc_readmail=$?
    fi
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "[runloop] readmail.py 结束，退出码 $rc_readmail，耗时 ${duration}s"
    
    echo "-------------------------------------------------------------------------------"
    
    # 倒计时
    if [ $INTERVAL -gt 0 ]; then
        countdown $INTERVAL
    fi
done