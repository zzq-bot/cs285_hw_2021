# Set up

## Environment Set up in Windows

**Remark:** 报错GLEW initalization error: Missing GL version，未解决。可以运行，但是无法record视频数据。

make sure that anaconda is installed.

### 1、VS Build Tools

安装visual studio 2019

<img src="C:\Users\19124\Desktop\RL\cs_285\cs285_hw_2021\pics\1.png" style="zoom:50%;" />

选中内容如图所示，进行安装。

（如果已经安装过visual studio，可以点击修改）

### 2、Mujoco

在官网[Download (roboti.us)](https://www.roboti.us/download.html)下载 mujoco200 win64

在user/username/路径下创建 .mujoco文件夹，将下载好的文件解压，并重命名为mujoco200

在官网 License page [License (roboti.us)](https://www.roboti.us/license.html) 下载mjkey.txt放入mujoco200的bin目录下。

<img src="C:\Users\19124\Desktop\RL\cs_285\cs285_hw_2021\pics\2.png" style="zoom:50%;" />

环境变量修改，参照博客：

[(20条消息) win10安装mujoco200,mujoco_py2.0.2.9,gym_xiyuChen的博客-CSDN博客](https://blog.csdn.net/weixin_44377470/article/details/104928067)

最终效果如图：

<img src="C:\Users\19124\Desktop\RL\cs_285\cs285_hw_2021\pics\3.png" style="zoom:33%;" />

<img src="C:\Users\19124\Desktop\RL\cs_285\cs285_hw_2021\pics\4.png" style="zoom:50%;" />

### 3、虚拟环境创建

基本按照installation.md文档要求即可。

但需注意在 `pip install -r requirements.txt`时要查看requirements.txt中相关包的版本。

在安装mujoco_py时，使用 `pip install mujoco_py==2.0.2.8`

