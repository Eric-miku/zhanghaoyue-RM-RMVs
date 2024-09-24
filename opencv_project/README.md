使用 C++ 和 OpenCV 进行图像处理
====
# 环境配置
## C++、Cmake和OpenCV库的配置
- 根据任务书安装OpenCV库
- 不确定以前是否安装，检查版本号
![检验C++和Cmake配置]("resources/process_picture_1.png")
## 构建项目目录
![构建项目目录]("resources/process_picture_2.png")
# CMakeLists.txt的编写
## 遇到困难
- 找不到OpenCV库
![问题]("resources/process_picture_3.png")
## 解决尝试
- 检查OpenCV库是否完整
- 查看include文件夹和local/include文件夹中是否有OpenCV4
- 查看群中相近的问题
- 问询AI
## 最终解决
- 点击左下方的生成
## 问题原因推测
- 之前一直直接点击右上方的三角形，应该是使用VScode中的编译器，但是使用了Cmake来配置项目，链接OpenCV库，因而要使用
<br>cd opencv_project/build
<br>cmake ..
<br>make -j
<br>./OpenCV_Project

# 主程序编写
## 图像颜色空间转换
### 任务目标
- 转化为灰度图
- 转化为 HSV 图片
### 遇到的问题
- 图片无法显示——waitkey()中设成了0。
### 任务结果
## 应用各种滤波操作
- 应用均值滤波
- 应用高斯滤波
- 无任何困难
## 特征提取 
### 提取红色颜色区域——HSV 方法
- 遇到问题：mask无论怎样处理都是全黑
- 解决：删除build，重新构建build
- 原因分析：代码迭代太多，导致build中产生了未知的干扰
## 图像绘制和tuxiang处理部分
- 用AI写的