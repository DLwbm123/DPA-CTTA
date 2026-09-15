# 未运行与证据边界

- 尚未执行真实 checkpoint GPU smoke、目标 RGB/mask 读取、正式 70 条轨迹或 GPU 后台队列。
- 程序化 CPU 数学、轻模型和随机权重完整 ResUNet 检查不能代替外部代码 review、GPU 验收或效果证据。
- 当前 private 注册解析只读取原登记 JSON，70 条 metadata dry-run 不解码像素。正式预算 136,570 评分/Adam/backward、1,112,070 forward、0 VJP；每实际设备 smoke 32 Adam/backward、260 forward。
- 最多 3 worker，每卡一个、每 worker 2 CPU 线程；单轨迹6h、墙钟72h、active-worker96h、新输出6GiB是停止上限，不是耗时预测。
- 没有新增源数据/代理/原型访问、源训练、其他 checkpoint、Polyp、层搜索、参数网格或路线组合。
- 没有生成外部 review pass。设备资源选择与代码 review 是不同条件；实际授权材料不公开。
- 历史 EIO 现象仍保留“本次未重现，根因未知”的边界；本轮回归通过不消除该历史。
