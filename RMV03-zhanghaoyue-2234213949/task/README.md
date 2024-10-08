# 能量机关识别与预测
## 总结
- 我在做这个项目时，除去下面提到的困难外，还遇见许多小难点，但可以保证项目主体都是我和AI一起完成的，在拟合phi时我也看了看别人的代码，发现也存在和我一样的问题，多谢组长最后放宽标准
## ceres库的安装
### Ubuntu22.04 安装ceres-solver，cmake编译报错有tbb_stddef.h
- tbb_stddef.h文件缺失
-  解决查看的链接[网页链接](https://blog.csdn.net/qq_45999722/article/details/129267563?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-129267563-blog-130683251.235%5Ev43%5Epc_blog_bottom_relevance_base7&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-129267563-blog-130683251.235%5Ev43%5Epc_blog_bottom_relevance_base7&utm_relevant_index=2)
### 安装时报错
- 关闭测试项目来解决
### Cmakelist.txt中链接ceres库
- 自己写时报错，通过AI修复时发现是代码的前后顺序错了
- 关键修复：
- target_link_libraries 的目标：现在使用了正确的目标名称 task，确保在链接 Ceres 时不会出错。
- 包含 Ceres 头文件：将 include_directories(${CERES_INCLUDE_DIRS}) 合并到 include_directories 中，这样 Ceres 的头文件路径被正确添加到编译路径中。
- aux_source_directory：这个命令将 ./src 目录下的所有源文件添加到 ALL_SRCS 变量中，确保源文件可以正确包含在可执行文件中。
## 对R标和锤子扇叶的识别
### 轮廓混为一体
- 通过输出检测，发现只有一个轮廓，推测此论廓为窗口，显示轮廓发现确实如此
- 将RETR_EXTERNAL换为RETR_TREE
### 图像处理
- R标中心内轮廓不容易筛除
- 使用形态学操作的组合
- 膨胀和腐蚀的结合：首先应用腐蚀，然后再应用膨胀，这种操作叫做开运算（Opening），可以去除小的噪声，同时保留较大的轮廓。
```
C++
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));  // 结构元素
    erode(binary, binary, element);   // 腐蚀操作
    dilate(binary, binary, element);   // 膨胀操作
```
### 对轮廓进行筛选
- 发现边界框长宽比不固定
![截图 2024-10-04 22-18-22](https://github.com/user-attachments/assets/4f6d72bf-ccc3-48e8-932b-05db29bfa919)
- 且在识别锤子时易与其他扇叶混淆，于是改用轮廓面积来筛选
### 计算中心
- “R”中心容易计算
- 锤子轮廓面积大，且在旋转中，用倾斜矩形框找到四个定点来计算
### 计算角度
## 对转速的拟合
### 基本步骤
- 1.定义每个参数块。参数块就是简单的向量，也可以是四元数、李代数等特殊的结构。
- 2.定义残差块的计算方式。残差块对参数块进行自定义计算，返回残差值，然后求平方和作为目标函数的值。
- 3.定义雅可比的计算方式。
- 4.把所有的参数块和残差块加入Ceres定义的Problem对象中，调用Solve函数求解
### 拟合过程
#### 第一次
- omega几乎没变化，对代码检查后也没问题
- 发现是时间出问题了，改成相对时间得以解决
#### 第二次拟合
- omega和fai都十分精确，显示收敛，但A与b与真实值相差较大
- 通过x-y图像显示，angle随时间的变化正常，但speed随时间的变化呈散点图
- ![截图 2024-10-05 18-38-00](https://github.com/user-attachments/assets/ded67267-3a39-4747-8348-31dae5cafe87)
![截图 2024-10-05 18-41-10](https://github.com/user-attachments/assets/2242af33-cac7-48fe-9f66-d41e880e740c)
- 推测原因
- 1.在获取angle时，看似平滑，但是求速度时因为除的dt很小，导致得到的speed成散点图，但b拟合结果为负！！！
- 2.由式子可以看出speed应为正值，估计是旋转方向问题
- 3.在识别时发现轮廓框在轻微晃动，可能导致角度噪声偏大
### 第三次拟合
- 确实是数据的问题
- 通过更改对锤子扇叶的质心识别，用轮廓来得到质心
- ![截图 2024-10-05 20-00-06](https://github.com/user-attachments/assets/28b39233-7ae2-48ef-aaec-29f942acbdf7)
- 观察图像发现存在上截止
### 第四次拟合
- 仅有phi不能很好的拟合，同时A会有反值的情况：
A : -0.793367
omega : 1.88407
phi : 1.05193
b : 1.2942
- 而且拟合的稳定性不高
- AI分析
- 在对风车的转速公式 Acos⁡(ωt+ϕ)+bAcos(ωt+ϕ)+b 进行拟合时，拟合的成功与失败可能受到多种因素的影响。以下是几个可能的原因：
#### 1. 初始值选择问题
-     你的任务要求初始值与真实值的差距大于1，如果初始值偏差过大，可能会导致优化算法无法找到全局最优解或陷入局部最优解。
-     拟合问题中的非线性优化算法（如Levenberg-Marquardt）对初始值非常敏感，初始值不佳可能会导致收敛困难。
#### 2. 数据质量问题
-     如果用于拟合的数据存在噪声或异常点，可能会干扰拟合的效果，导致拟合失败。
-     图像中的运动模糊、光照变化或者特征点识别不准确（例如风车中心和锤子位置的检测不准），都可能影响你提取到的角度数据，使得拟合结果不稳定。

#### 3. 优化算法参数设置
-    优化算法中的参数（如最大迭代次数、容忍度等）设置不合理，可能会影响拟合的效果。如果容忍度过高，可能会导致算法过早停止，从而导致拟合失败。
#### 4. 曲线拟合的非线性特征
-     cos⁡cos 函数是周期性的，如果初始的拟合结果使得参数跳到了错误的周期上，可能会导致失败。特别是当 ωt+ϕ的初始估计导致该值接近 2π 或者周期边界时，算法可能难以调整回正确的周期。
#### 5. 帧处理一致性问题
-     你需要确保每一帧都被处理，且不允许跳帧。如果某些帧被跳过或者数据不一致，可能会影响拟合的连续性和准确性，尤其是当帧率波动较大时。
#### 6. 风车的复杂运动
-     由于风车的运动是随机的，可能存在一些复杂的运动模式，这些模式可能不完全符合拟合的假设。你需要确保在拟合过程中，检测的运动特征足够准确且连续。
- 解决建议：
-     尝试选择更接近真实值的初始值，逐步调整初始值与真实值的偏差范围。
-     检查数据提取的准确性，尤其是R标记和锤子的位置识别是否足够稳定。
-     增加对数据的平滑处理，减少噪声的影响。
-     调整优化算法的参数，尤其是最大迭代次数和容忍度。
### 第五次拟合
- 数据质量已优化，排除风车的复杂运动和帧处理一致性问题，不考虑初始值选择问题和优化算法参数设置
- 最直接的优化是调整因曲线拟合的非线性特征产生的失败拟合
## 暂停一下，分析分析
- 1.据以往实验数据来看，A、omega、b的拟合都非常成功，详情可见第四次拟合数据展示，问题就出现在phi上；
- 2.不知道出于什么原因，A与omega的值有时候会取负；感觉跟上面的phi的不准有关；
- 3.不能先确定A、omega、b，再确定phi，因为确定A、omega、b时，用的是A、omega、b、phi四个估计值，他们相互影响。退一步讲，就算是这样做，由历史数据可见，phi依然不准确。
### 最终成功拟合版本
#### 其中一次结果
```
======================
三个参数拟合成功
第1次拟合成功
A : -0.808179
omega : 1.88486
phi : 1.12297
b : 1.31007
======================
第2次拟合失败即将开始重新拟合
A : 0.831364
omega : -1.87032
phi : 2.59823
b : 1.32209
======================
三个参数拟合成功
第2次拟合成功
A : -0.769223
omega : -1.85025
phi : 1.34274
b : 1.29957
======================
三个参数拟合成功
第3次拟合成功
A : -0.793492
omega : 1.96136
phi : -4.50834
b : 1.35871
======================
三个参数拟合成功
第4次拟合成功
A : 0.779403
omega : 1.89808
phi : 1.38796
b : 1.31712
======================
第5次拟合失败即将开始重新拟合
A : -1.59467
omega : -0.10648
phi : 4.52283
b : 0.50967
======================
第5次拟合失败即将开始重新拟合
A : -0.0652517
omega : 72.9902
phi : -59.096
b : 1.19414
======================
三个参数拟合成功
第5次拟合成功
A : -0.786091
omega : 1.84931
phi : -0.572998
b : 1.29678
======================
三个参数拟合成功
第6次拟合成功
A : 0.782423
omega : 1.95379
phi : 2.83652
b : 1.32827
======================
三个参数拟合成功
第7次拟合成功
A : -0.771116
omega : 1.86866
phi : -1.07476
b : 1.28114
======================
第8次拟合失败即将开始重新拟合
A : -0.090658
omega : -21.7903
phi : 12.3181
b : 1.28895
======================
三个参数拟合成功
第8次拟合成功
A : 0.775809
omega : 1.90628
phi : 1.0838
b : 1.32611
======================
三个参数拟合成功
第9次拟合成功
A : -0.814044
omega : 1.89352
phi : -0.895495
b : 1.33358
======================
第10次拟合失败即将开始重新拟合
A : 0.837583
omega : -1.87296
phi : 1.74572
b : 1.29435
======================
三个参数拟合成功
第10次拟合成功
A : 0.765818
omega : 1.8737
phi : -0.147513
b : 1.29664
7.385
```
- 其中一次成功拟合的可视化截图
![截图 2024-10-06 11-41-28](https://github.com/user-attachments/assets/564330bc-14af-4bc4-852c-f412e29e0115)
#### 成功原因
- 引用组长的原话：
```
喵：发现是重合的就对
喵：这无所谓了，相位涉及一些换算，你们可能不一定能推出来
```
