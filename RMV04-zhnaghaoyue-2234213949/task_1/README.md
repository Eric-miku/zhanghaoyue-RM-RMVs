# 基础层次
## 总结
- 这是我完成的基础层次的README，并未在include中创建头文件，但是声明了灯条类，对图像处理的部分是我自己查文档、问AI写的，类的部分主要是参考了[这篇SCDN文章](https://blog.csdn.net/qq_64659054/article/details/126674590?ops_request_misc=%257B%2522request%255Fid%2522%253A%25227D57FF1E-18F4-445A-982D-43F7155E3677%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=7D57FF1E-18F4-445A-982D-43F7155E3677&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-12-126674590-null-null.142^v100^pc_search_result_base5&utm_term=opencv%E8%AF%86%E5%88%AB%E7%81%AF%E6%9D%A1&spm=1018.2226.3001.4187)，同时阅览了SCDN上的其他相关文档。

## 缺少 GStreamer 插件
- 通过以下指令安装
- sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
## 构建项目的根目录
## 实现
### 步骤
#### 图像预处理：
- 从视频帧中获取图像。
- 转换为灰度图像，并根据装甲板的颜色或亮度特征进行二值化处理。
#### 装甲板特征识别：
- 使用 cv::findContours() 检测轮廓，并根据装甲板的矩形特征或颜色形态来识别它们。
- 通过轮廓的大小和形状（例如矩形或长方形）来过滤出装甲板。
- 使用 cv::minAreaRect() 来获取装甲板的倾斜矩形，并通过该矩形的质心、边界等信息标注其位置。
#### 标注像素位置与现实位置：
- 使用 OpenCV 绘制矩形或圆圈，标注装甲板的像素位置。
- 如果视频中有已知的标尺或尺寸信息，可以通过简单的换算将像素位置转换为现实意义的位置。
### 面临的困难
#### 视频选择
- 群中发了两个视频，一个装甲板为red，一个装甲板为blue。
- ![截图 2024-10-12 21-50-12](https://github.com/user-attachments/assets/e7e6d6fb-95f0-4948-a048-2756ca22b3ec)
- red 视频中曝光太高了，无论怎样处理，都无法减少无关轮廓，应该选blue
#### HSV范围调整
- 一直无法消除灯条中间的断开部分，试过很多参数都不行
![截图 2024-10-12 22-36-35](https://github.com/user-attachments/assets/fa442548-cd6b-43b8-bf42-21bd569f600c)
![截图 2024-10-12 22-35-26](https://github.com/user-attachments/assets/6ad079bc-b24b-489f-808e-3de73218ad4c)
- 最终采用灰度进行显示轮廓，效果很好
#### 关于灯带的匹配
- 最开始考虑角差、长度差比等各种参数以及拟合轮廓的方法（矩形或椭圆）
- 但是怎样调都会有错误识别的帧存在，错误情况包括多出识别方框，无法识别目标方框等等
- 最终采用选取轮廓面积最大的两个轮廓，效果非常好，得到的装甲板框非常稳定，且不需要考虑拟合的选择
#### 装甲板框90度旋转问题
- RotatedRect类的angle变量的范围为0-180，长轴在识别时角度接近180,会产生突变
- 通过两个角度相减找到异常，人为的加180度
![截图 2024-10-13 01-30-12](https://github.com/user-attachments/assets/9c533bd0-bca4-439e-afb0-94ee06f05e6e)

