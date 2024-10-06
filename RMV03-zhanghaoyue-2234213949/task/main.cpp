/*
项目名称：能量机关识别与预测 
项目作者：章昊越
作者邮箱：2446190505@qq.com
*/

/*
此版本改动：
取消分步拟合，增加可视化，将判断的条件改为对A，omega和b
对可视化图像观察发现在第二次、能收敛的就已经能收敛了，估计数据量在600到900之间
增加对异常数据的剔除
对代码整体进行优化
*/

/*
此版本细节详情：
1.对是否除以1000的处理，windmill.cpp中进行了“/1000”，所以在得到图像时未处以1000
2.识别条件改为了仅判断A，omega和b
3.增加对异常数据的处理，避免局部最优解
4.在可视化时，为保证能看到拟合后的曲线，我令程序暂停了1秒
5.在统计数据时，异常数据所消耗的时间也会被统计
*/

#include "windmill.hpp"
#include<iostream>
#include <ceres/ceres.h>

using namespace std;
using namespace cv;
// 计算残差的结构体
struct WindmillCostFunctor {
    WindmillCostFunctor(double t, double speed) : t(t), speed(speed) {}
    template <typename T>
    bool operator()(const T* const A, const T* const omega, const T* const phi, const T* const b, T* residual) const {
        //观测值为负，预测值为正，对预测值取负，排除方向影响
        residual[0] = T(speed) + (A[0] * cos(omega[0] * T(t) + phi[0]) + b[0]);
        return true; 
    }
    double t;           // 时间
    double speed;       // 测量到的角速度  
};

//计算角度
double calculateAngle(const Point2i& , const Point2i& );
//处理风车图像，得到用于计算的角度，引用了calculateAngle(const Point2i& , const Point2i& )
void processImage(const Mat ,double&);
/*
可视化
ShowFitting（）创建窗口、绘制数据点和拟合曲线
FitFunction（）通过x轴来计算y轴，A、omega、phi、b为计算时的常量
*/
void ShowFitting(const std::vector<double>& x_data, const std::vector<double>& y_data, double A,double  omega,double phi,double b);
double FitFunction (double x, double A,double  omega,double phi,double b);

int main()
{
    double pai=3.141592;

    double t_sum = 0;
    const int N =10;
    for (int num = 0; num < N; num++)
    {   
        int flag=0;
        cout<<"======================"<<endl;   //分界线
        std::vector<double> angles;     // 存储角度
        std::vector<double> times;      // 存储时间
        std::vector<double> speeds;     //存储速度

        //对参数预测值初始化，及对循环中辅助循环进行的变量进行初始化
        double  A = 1.785, omega = 0.884,phi = 1.24 , b = 0.305;
        ceres::Problem problem;
        int count = 0;                  //帧数计数
        double old_t = 0; 
        double old_angle = 0;
 
        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());//t单位为毫秒
        double t_start = (double)t.count();     //定义开始时间，为WindMill类作初始化作准备
        int64 start_time = getTickCount();      // starttime，计算程序运行时间的开始
        WINDMILL::WindMill wm(t_start);         //风车初始化
        Mat src;                              
        while (1)
        {   
            double now_angle = 0;
            t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            //获取风车图像
            double now_t = (double)t.count();
            src = wm.getMat(now_t); 
            //处理风车图像，得到用于计算的角度
            processImage(src,now_angle);
            //显示风车图像
            imshow("windmill", src);

            //数据记录
            now_t= (now_t- t_start)/1000;   //获取相对时间
            
            if(count!=0){               
                old_t = times.back();
                old_angle = angles.back();
                if(count>=4){               //对所得角度平滑处理
                     now_angle = 0.5 * now_angle + 0.3 * angles[count-1] + 0.2 * angles[count-2];
                }
                double speed = (( now_angle - old_angle ) *3.1415)/ (( now_t - old_t )*180);
                speeds.push_back(speed);    // 记录速度
            }else{
                double speed = 1.3;
                speeds.push_back(speed);
            }
            angles.push_back(now_angle);    // 记录角度
            times.push_back(now_t);         // 记录时间
            count++;                        //计数器更新

            //每300帧拟合一次，排除第一次
            if(count % 300 == 0 && count !=0 ){
                for (size_t i = 0; i < speeds.size(); i++) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<WindmillCostFunctor, 1, 1,1,1,1>(
                        new WindmillCostFunctor(times[i], speeds[i])),
                    new ceres::CauchyLoss(0.5),
                    //nullptr,
                    &A, &omega, &phi, &b);
                }
                // 求解问题
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }

            //可视化展示
            ShowFitting(times,speeds,A,omega,phi,b);

            double m_A=abs(abs(A)-0.785)/0.785;
            double m_omega=abs(abs(omega)-1.884)/1.884;          
            double m_b=abs(abs(b)-1.305)/1.305;
            if(flag == 0 && m_A < 0.05 && m_omega < 0.05 &&m_b < 0.05 ){
                //&&m_phi<0.05
                //cout<<"第"<<num+1<<"次拟合成功"<<endl;
                cout<<"三个参数拟合成功"<<endl;
                cout<<"第"<<num+1<<"次拟合成功"<<endl;
                cout<<"A : "<<A<<endl<<"omega : "<<omega<<endl<<"phi : "<<phi<<endl<<"b : "<<b<<endl;
                flag=1;
                waitKey(1000);
                break;
            }

            //手动结束当前循环,或者到达帧数上界停止拟合
            if (cv::waitKey(1) >= 0||count==900) {
                break; 
            } 
        } 
        if(flag ==0){
            cout<<"第"<<num+1<<"次拟合失败"<<"即将开始重新拟合"<<endl;
            cout<<"A : "<<A<<endl<<"omega : "<<omega<<endl<<"phi : "<<phi<<endl<<"b : "<<b<<endl;
            num--;
        }
        //计算程序运行时间的结束
        {   int64 end_time = getTickCount();
            double elapsed_time = (end_time - start_time) / cv::getTickFrequency(); // 计算经过的时间（秒）
            t_sum += elapsed_time;}
    } 
    //求平均时间
    std::cout << t_sum / N << std::endl;
    cout<<"over";
    return 0;
}

// 计算两个点之间的角度
double calculateAngle(const Point2i& center,Point2i& point) {
    return atan2(point.y - center.y, point.x - center.x) * 180 / CV_PI;
}

// 处理图像
void processImage(Mat src,double& currentAngle) {
    Mat gray, binary;
    Point2i rCenter,hammerCenter;
    
    cvtColor(src, gray, COLOR_BGR2GRAY);// 转换为灰度图像
    
    //GaussianBlur(gray, gray, Size(5, 5), 0);// 应用高斯模糊以减少噪声
    
    threshold(gray, binary, 50, 255, THRESH_BINARY_INV);//转化为二值化图像
    //
    Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
    erode(binary, binary, element);   // 腐蚀
    dilate(binary, binary, element);   // 膨胀

    // 寻找轮廓
    vector<vector<Point>> contours;
    vector<vector<Point>> filteredContours;
    
    findContours(binary, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(255, 255, 0); // 绿蓝色
        drawContours(drawing, contours, (int)i, color, 2, 8); // 绘制轮廓
    }
    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour); // 计算轮廓的边界框
        //筛选锤子外轮廓
        if (contourArea(contour)<10000 && contourArea(contour)>2500) {
            //计算中点
            // 计算轮廓的矩
            cv::Moments mu = cv::moments(contour);
            // 计算质心
            cv::Point2f Center(mu.m10 / mu.m00, mu.m01 / mu.m00);
            hammerCenter=Center;
        // 在输出图像上绘制轮廓和质心
        cv::drawContours(src, contours, -1, cv::Scalar(0, 255, 0), 2);
        cv::circle(src, hammerCenter, 5, cv::Scalar(255, 0, 0), -1); // 红色点表示质心
            //绘制
            putText(src, "hammer", hammerCenter, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        } 
        //筛选“R”标
        if (contourArea(contour)<500) {
            //计算中点
            Point2i center = (boundingBox.br() + boundingBox.tl()) * 0.5;
            rCenter = center;
            //绘制
            circle(src, rCenter, 5, Scalar(255, 0, 0), 2);  // 在图像中标记 "R" 标记的中心
            rectangle(drawing, boundingBox.tl(), boundingBox.br(), Scalar(255, 0, 0), 3);
            putText(src, "R", rCenter, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
    }
    imshow("Contours and Bounding Boxes", drawing);
    // 计算当前风车的角度
    if (rCenter != Point2i(-1, -1) && hammerCenter != Point2i(-1, -1)) {
        currentAngle = calculateAngle(rCenter, hammerCenter);
    }
}

void ShowFitting(const std::vector<double>& x_data, const std::vector<double>& y_data, double A,double  omega,double phi,double b) {
    cv::Mat image = cv::Mat::zeros(1600, 1800, CV_8UC3); // 创建一个黑色背景图像

    // 绘制数据点
    for (size_t i = 0; i < x_data.size(); ++i) {
        cv::circle(image, cv::Point(static_cast<int>(x_data[i] * 200 + 300), static_cast<int>(500 - y_data[i] * 100)), 1, cv::Scalar(0, 255, 0), -1);
    }

    // 绘制拟合线
    for (double x = 0; x <= 6; x += 0.1) {
        double y = FitFunction(x, A,omega,phi,b);
        cv::circle(image, cv::Point(static_cast<int>(x * 200 + 300), static_cast<int>(500 - y * 100)), 2, cv::Scalar(0,0 , 255), -1);
    }

    // 显示图像
    cv::imshow("Fitting Result", image);
    cv::waitKey(1); // 更新显示
}

double FitFunction(double x, double A,double  omega,double phi,double b) {
    return -(A * cos(omega * x + phi) + b); // 线性模型示例
}