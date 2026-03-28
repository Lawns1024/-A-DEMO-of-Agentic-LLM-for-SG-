# CEVI IoT MAC 图表生成器

该脚本基于 `cevi-iot-policy.csv` 生成可插入论文的吞吐量/碰撞率图表。

## 使用方法

1. 先运行 ns-3 仿真，确保生成 `cevi-iot-policy.csv`。
2. 安装依赖：
   - `pip install -r requirements.txt`
3. 运行绘图脚本：
   - `python plot_cevi_iot.py --csv /path/to/cevi-iot-policy.csv`

输出：
- `cevi-iot-metrics.png`
- `cevi-iot-metrics.pdf`

## CSV 列说明

- `time_s`：仿真时间（秒）
- `node_id`：节点 ID
- `state`：策略状态
- `cw_min` / `cw_max`：退避窗口
- `throughput_mbps`：吞吐量（Mbps）
- `collision_rate`：碰撞率（0~1）
