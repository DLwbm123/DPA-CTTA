# r10源开发结果

开发门：NOT PASSED。不是独立患者/真实目标验证。

|种子|FULL−DIRECT pp|FULL−SOURCE pp|FULL−NO_UPDATE pp|FULL−EPISODIC pp|FULL−RANDOM pp|门|
|---|---:|---:|---:|---:|---:|---|
|10223|0.672651|2.310393|NA|NA|NA|False|
|10224|0.452563|1.986270|NA|NA|NA|False|

种子10223 FULL−独立训练固定DIRECT残差对照: 0.034716pp。

种子10224 FULL−独立训练固定DIRECT残差对照: -0.051130pp。

检查点仅据开发集选取；bootstrap以内容为单位合并顺序和扰动，仅描述性。两个种子共享开发集，不作为独立队列。未通过门时保留负结果，不修改门。历史接口失败与工程重试分别归档，见各轮协议和运行回执。
