# r12源开发结果

开发门：PASS。不是独立患者/真实目标验证。

|种子|FULL−DIRECT pp|FULL−SOURCE pp|FULL−NO_UPDATE pp|FULL−EPISODIC pp|FULL−RANDOM pp|门|
|---|---:|---:|---:|---:|---:|---|
|10223|1.152889|2.790631|NA|NA|NA|True|
|10224|1.189234|2.722941|NA|NA|NA|True|

种子10223 FULL−独立训练固定DIRECT残差对照: 0.370593pp。

种子10224 FULL−独立训练固定DIRECT残差对照: 0.370095pp。

检查点仅据开发集选取；bootstrap以内容为单位合并顺序和扰动，仅描述性。两个种子共享开发集，不作为独立队列。未通过门时保留负结果，不修改门。历史接口失败与工程重试分别归档，见各轮协议和运行回执。
