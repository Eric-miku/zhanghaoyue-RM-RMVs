//打算分步拟合

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
    double t;    // 时间
    double speed; // 测量到的角速度  
};
struct WindmillCostFunctor_ {
    WindmillCostFunctor_(double t, double speed,double A,double omega,double b) : t(t), speed(speed) {}
    template <typename T>
    bool operator()( const T* const phi,  T* residual) const {
        //观测值为负，预测值为正，对预测值取负，排除方向影响
        residual[0] = T(speed) + (A * cos(omega * T(t) + phi[0]) + b);
        return true; 
    }
    double t;    // 时间
    double speed; // 测量到的角速度
    double A;
    double omega;
    double b;
};



double calculateAngle(const Point2i& , const Point2i& );
void processImage(const Mat ,double&);

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
        cout<<"======================"<<endl;
        std::vector<double> angles;     // 存储角度
        std::vector<double> times;      // 存储时间
        std::vector<double> speeds;

        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());//t单位为毫秒
        double t_start = (double)t.count(); //定义开始时间，为WindMill类作初始化作准备
        

        double  A = 1.785, omega = 0.884,phi = 1.24 , b = 0.305;
        //A = 0.785;w = 1.884;fai = 1.65;b = 1.305;   
        ceres::Problem problem;
        int count = 0;
        double old_t = 0; 
        double old_angle = 0;
 
        int64 start_time = getTickCount();  // starttime，计算程序运行时间的开始
        WINDMILL::WindMill wm(t_start);     //风车初始化
        Mat src;                            //每帧要处理的图片
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

            now_t= (now_t- t_start)/1000;
            if(count!=0){
                //数据记录
                old_t = times.back();
                old_angle = angles.back();
                if(count>=4)                //平滑处理
                {
                    for(int i=0;i<3;i++)
                    {
                        now_angle+=angles[count-i-2];
                    }
                    now_angle/=4;
                }
                double speed = (( now_angle - old_angle ) *3.1415)/ (( now_t - old_t )*180);
                speeds.push_back(speed);  // 记录速度
            }else{
                double speed = 1.3;
                speeds.push_back(speed);
            }
            angles.push_back(now_angle);  // 记录角度
            times.push_back(now_t);        // 记录时间
            count++;    

            ShowFitting(times,speeds,A,omega,phi,b);

            
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

                //options.logging_type=ceres::SILENT; 
                //options.linear_solver_type = ceres::DENSE_SCHUR;
                //options.minimizer_progress_to_stdout = true;
                //options.max_num_iterations = 2000; // 增加最大迭代次数
                // options.function_tolerance = 1e-6;  // 更严格的收敛标准
                // options.max_num_iterations = 2000;
                // options.function_tolerance = 1e-9;
                //options.gradient_tolerance = 1e-6;

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                cout<<"A : "<<A<<endl<<"omega : "<<omega<<endl<<"phi : "<<phi<<endl<<"b : "<<b<<endl;

            }

            double m_A=abs(abs(A)-0.785)/0.785;
            double m_omega=abs(abs(omega)-1.884)/1.884;
            
            double m_b=abs(abs(b)-1.305)/1.305;
            if(flag==0&&m_A<0.03&&m_omega<0.03&&m_b<0.03){
                //&&m_phi<0.05
                //cout<<"第"<<num+1<<"次拟合成功"<<endl;
                cout<<"三个参数拟合成功"<<endl;
                A=abs(A);
                omega=abs(omega);
                b=abs(b);
                flag=1;
                cout<<"A : "<<A<<endl<<"omega : "<<omega<<endl<<"phi : "<<phi<<endl<<"b : "<<b<<endl;

            }

            if(flag==1){
                
                    
                ceres::Problem problem_;
                    
                //int count_=0;
                while(1){
                        while(phi>pai){
                            phi-=2*pai;
                        }
                        while(phi<0){
                            phi+=2*pai;
                        }
                    //count % 300 == 0 && count !=0
                    for (size_t i = 0; i < speeds.size(); i++) {
                    problem_.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<WindmillCostFunctor_,1,1>(
                            new WindmillCostFunctor_(times[i], speeds[i],A,omega,b)),
                        new ceres::CauchyLoss(0.5),                        &phi);
                    }
                    // 求解问题
                    ceres::Solver::Options options_;
                    options_.linear_solver_type = ceres::DENSE_QR;
                    ceres::Solver::Summary summary_;
                    ceres::Solve(options_, &problem_, &summary_);
                    cout<<"A : "<<A<<endl<<"omega : "<<omega<<endl<<"phi : "<<phi<<endl<<"b : "<<b<<endl;

                    //count_++;
                }
                    
                

            
                double m_phi=abs(phi-1.65)/1.65;
                if(m_phi<0.05){
                    cout<<"第"<<num+1<<"次拟合成功"<<endl;
                }
            }
            

            //手动结束当前循环
            if (cv::waitKey(1) >= 0||(count==1500&&flag==0)) {
                break; 
            } 
        } 
        if(flag ==0){
            cout<<"第"<<num+1<<"次拟合失败"<<endl;
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
    for (double x = -3; x <= 3; x += 0.1) {
        double y = FitFunction(x, A,omega,phi,b);
        cv::circle(image, cv::Point(static_cast<int>(x * 200 + 300), static_cast<int>(500 - y * 100)), 1, cv::Scalar(255, 0, 0), -1);
    }

    // 显示图像
    cv::imshow("Fitting Result", image);
    cv::waitKey(1); // 更新显示
}

double FitFunction(double x, double A,double  omega,double phi,double b) {
    return A * cos(omega * x + phi) + b; // 线性模型示例
}