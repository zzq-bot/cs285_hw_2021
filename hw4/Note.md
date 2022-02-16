每次target计算的是：

先算出$\delta_u = s_{t+1}-s_t$，

然后用memory中的delta_mean, delta_std 去normalize$(\delta_u-mean)/std$ -> $\delta_n$

所以预测出来的就是$\hat \delta_n$，用$\hat \delta_n,\delta_n$去计算loss（用normalize过的$s_t,a_t$去预测）

用$\hat \delta_n*std+mean$去得到$\hat s_{t+1}$
