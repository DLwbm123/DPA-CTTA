# run12 配对gamma任务风险约束（运行前冻结）

diag11预注册触发：1121项gamma误差改善条件中32.56%分割下降，超过25%，因此检验参数误差目标与分割风险对齐。冻结run01base和run06外观/RGB，复用run10 Gamma结构及features，不加载run10训练权重。C与R相同零残差初始化、同一source fit图像/光度随机样本、每源1200步batch2（clean+aug共4），seed92512，AdamW .001 wd.0001。

C损失仅loggamma MSE。R损失=MSE+.1*mean(L_R)+mean(relu(L_R−L_I))，I为冻结run06输出。L为每图OD/OC soft-Dice loss平均：1−(2sum(prob*y)+1e−6)/(sum(prob)+sum(y)+1e−6)，prob=sigmoid(−16*预测distance)。源标签只用于训练R/源验证；冻结base保留输入梯度。该新增项同时含任务均值与超额风险，不能把差异独归于hinge；单一固定权重无网格。C为主线机制控制，同数据/预算。

所有五源训练后source dev clean/fixed_appearance/color/blur验证I/C/R/ZERO。预设R−I两已知扰动源等权≥.1pp、每源非负，R−C源等权≥0，clean R−I≥−.05pp，扰动pooled p10(R−I)≥−.1pp（风险路线要求比既有−1更严格），三已知条件gamma MSEratio(R/I)≤.8。全部过才五源R/C目标各一次，I/ZERO复用，不挑源/不改旧门/不目标选参。任何失败即不目标。

源dev多次开发/base源暴露，非独立验证；无跨图状态，不称持续成功。C/R同批次但两目标不同，完整公开阴性结果。3小时硬时钟，TERM提前300秒KILL60秒，2GBsoft3GBhard预计<500MB；总新增<50GB45GB预停。唯一30min监测，11:37用户窗口汇总不强停健康任务。
