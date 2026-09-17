# 当前 R6 CPU 验收覆盖

本次候选 `c94fff7c05cec38d541c441b62a9381ee46ba076`：两端分别完整166/166，失败/错误/跳过0、退出码0；原参考8/8的新运行与3/3比较方法分别保存。共同输入240组、gate对照402组，容差沿用既有冻结值。完整结果与成本见 [ORIGINALS_ACCEPTANCE.json](ORIGINALS_ACCEPTANCE.json)，新日志位于 logs/originals-redelivery/。

本轮补入科学配置后重新执行全部R6 23、R5 31、既有回归112项；没有拿旧d875f20日志替代。后续文档发布SHA未重新跑套件。两端均CUDA未初始化、真实RGB/mask/checkpoint/source读取0，服务器临时fixture位于专用NAS目录。原件及历史数学日志不改写。

既有各测试的性质覆盖和首失败/修正历史保持不变，完整历史说明见 [原覆盖文档](history/f9ec00bb/TEST_COVERAGE.md)。额外logs/host-scalar-probe.log对应旧438ff073候选，logs/final-host-scalar-probe.log对应d875f20；两者均未在c94fff7重跑，不改标为本次结果。完整套件中的host/标量/ledger检查已重新运行。

新增 check_originals_cpu.py 验证原MANIFEST和config字节绑定、矩阵/预算/执行关闭、错误science摘要拒绝；在共同程序化z/q上对比原参考与生产loss、逐像素梯度、权重、能量、倍率、seed及RNG；gate阈值从原提案取值独立计算。原参考通过import加载，未执行会覆盖历史日志的main。

成本边界：完整suite的B1 forward/backward/Adam hook计数并非全方法总调用量；runner预置的jacobian_vjp_calls=0不是继承回归的实测VJP总计，该总计未知。新小张量比较每端480次autograd求导，原参考新运行8次，无模型前向/Adam。CPU程序化故障注入日志不代表真实实验失败，也不能当真实实验receipt。
