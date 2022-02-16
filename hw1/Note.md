# 1、Behavior Cloning

在running loop中收集数据and训练agent

if itr==0，也即训练刚开始，我们利用收集好的expert data去做监督学习训练。else（在Dagger中），则用当前policy去收集数据，用expert policy去relabel每个state应该做什么action，用relabel的data再去监督学习。

bc_agent用mlp作为actor预测action。分为discrete和continuous情形。前者预测Categorical分布，后者预测dim-wise的gaussian分布（每个dim各自std）。

notice that在get action时，如果用sample是无法梯度反向传播的（rsample用了重参数技巧，可以）。但是最好直接用MLE的公式，根据forward得到的distribution和expert action，用-dist.log_prob进行计算，进行梯度下降。