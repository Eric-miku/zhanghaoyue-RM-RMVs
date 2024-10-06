//hpp不处1000
#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
#include "windmill.hpp"
#include <vector>
using namespace std;
using namespace cv;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;


std::chrono::milliseconds t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());//初始时间
double t0=0;


struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y):_x(x),_y(y){}
    
    // 残差的计算
    template <typename T>
bool operator()(const T* const abcd, T* residual )const
{
    // 使用模板类型T，而不是直接使用double     A0 A w fai的顺序对应abcd
    T jiaodu = cos(abcd[0] * _x + (abcd[1] / abcd[2]) * (ceres::cos(abcd[2]*T(t0)+abcd[3]) - ceres::cos(abcd[2] * (_x+T(t0) ) + abcd[3])) );
    // 计算残差
    residual[0] = T(_y) - jiaodu;
    // 返回true表示计算成功
    return true;
}
    // 观测点的x坐标
    const double _x;
    // 观测点的y坐标
    const double _y;
};
int main(int argc,char** argv)
{   
    double zongshijian;
    google::InitGoogleLogging(argv[0]);
    double A0Awfai[4]={0,0,0,0};
    for(int n=0;n<10;n++)
    {int N=1600;  // 数据个数
    std::chrono::milliseconds t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());//初始时间
    double abcd[4]={1.5,0.5,1.5,1.5};  //a b c d参数的估计值
 
    double  x_data[N],y_data[N];//数据
 
    cout<<"generating data: "<<endl;
    
    WINDMILL::WindMill wm(t0);
    cv::Mat templ;
    double data[100000];
    int cishu=0;
    int quedian=0;
  while (cishu<(N))
    {
        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        templ = wm.getMat((double)(t0+t.count()-t_start.count())/1000);//绘制的图
        
        //==========================代码区========================//
           
           
           
           std::vector<std::vector<cv::Point>> contour2s;
           Mat src1=templ.clone();
    cv::cvtColor(src1,src1,cv::COLOR_BGR2GRAY);
    cv::findContours(src1, contour2s, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 绘制blue色 bounding box
    cv::Mat src4=templ.clone();
       std::vector<double> areas;
    //计算框面积
    for (const auto& contour : contour2s) {
        double area = cv::contourArea(contour);
        areas.push_back(area);
    }

for(int i=0;i<5;i++)
{

    for(int j=i+1;j<6;j++)
    {

    if(areas[i]>areas[j])
    {
   std::swap(areas[i],areas[j]);
   std::swap(contour2s[i],contour2s[j]);

    }
    }
}

//识别R
        std::vector<cv::Point> contour3s=contour2s[0];
        cv::RotatedRect rect = cv::minAreaRect(contour3s);
        cv::Point2f points[4];
        rect.points(points); // 获取旋转矩形的四个顶点
        cv::Point2i yuanquan1=(points[0]+points[2])/2;//圆圈中心
        cv::circle(src4, yuanquan1, 20, (0,0,255), 2);//参数分别是（图片，左上角坐标，半径，颜色向量，布尔值（只有ture与false的区别））

//识别锤子
        cv::RotatedRect rect1 = cv::minAreaRect(contour2s[1]);
        cv::Point2f points1[4];
        rect1.points(points1); // 获取旋转矩形的四个顶点
        cv::Point2i yuanquan2=(points1[0]+points1[2])/2;//锤子整体正中心
        cv::Point2i yuanquan3=yuanquan2+(yuanquan2-yuanquan1)*0.5;//锤子整体正中心
        cv::circle(src4, yuanquan3, 10, (0,0,255), 2);//参数分别是（图片，左上角坐标，半径，颜色向量，布尔值（只有ture与false的区别））
    
    // 显示结果
    cv::imshow("动态图片", src4);


 
 
int x=yuanquan3.x-yuanquan1.x;
int y=yuanquan3.y-yuanquan1.y;
double r=sqrt(x*x+y*y);
double g=x/r;
double shijian=(double)(t.count()-t_start.count())/1000;
if(g!=0&&quedian%1==0){
x_data[cishu]=shijian;
double k=(g);
y_data[cishu]=(double)k;
//cout<<x_data[cishu]<<" "<<y_data[cishu]<<endl; //显示出收集的数据，最好不要点开，会拉慢运行速度
cishu++;
}

quedian++;

        //=======================================================//
                waitKey(1);
    }

    //构建最小二乘问题
    ceres::Problem problem;
    for(int i=0;i<N;i++)
    {
        /*向问题中添加误差项*/
        problem.AddResidualBlock(
            /*使用自动求导，模板参数：误差类型，输出纬度，输入纬度，数值参照前面struct中写法*/
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,4>(
                new CURVE_FITTING_COST(x_data[i],y_data[i])
            ),
            new ceres::CauchyLoss(0.5), // 这里设置 Huber 损失的 δ 参数 
            abcd//待估计参数
        );
    }
 
    //配置求解器
    ceres::Solver::Options options;//这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;//增量方程如何求解
    //options.minimizer_progress_to_stdout = true;//输出到cout
 
    ceres::Solver::Summary summary;//优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//记录时间
    
    //开始优化
    ceres::Solve(options,&problem,&summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//记录优化完的时间
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"数据收集消耗时间= "<<x_data[N-1]<<"s"<<" 拟合消耗时间 = "<<time_used.count()<<"s"<<" 一共消耗时间="<<x_data[N-1]+time_used.count()<<endl;
   
                if(abcd[2]<0)
                {
                    abcd[2]=-abcd[2];//w有时候被拟合成负的，大小不变；
                }
                while(abcd[3]>2)
                {
                    abcd[3]=abcd[3]-6.2831;
                }
                while(abcd[3]<0)
                {
                    abcd[3]=abcd[3]+6.2831;
                }
                 A0Awfai[0]=A0Awfai[0]+abcd[0],   A0Awfai[1]=A0Awfai[1]+abcd[1],   A0Awfai[2]=A0Awfai[2]+abcd[2],   A0Awfai[3]=A0Awfai[3]+abcd[3];
                 zongshijian = zongshijian+x_data[N-1]+time_used.count();
                 cout<<"真实 A0 A w fai = 1.305 0.785 1.884  1.81  "<<endl;
                 cout<<"测量所得到 = ";
                for ( auto a:abcd ) std::cout<<a<<" ";
                std::cout<<endl;
    }
    //输出结果
    cout<<"总时间 = "<<zongshijian<<"s"<<endl;
    cout<<"真实 A0 A w fai = 1.305 0.785 1.884  1.81  "<<endl;
    double A0=A0Awfai[0]/10,A=A0Awfai[1]/10,w=A0Awfai[2]/10,fai=A0Awfai[3]/10;
    cout<<"测得 A0 A w fai = "<<" "<<A0<<" " <<A<<" "<<w<<" "<<fai<<endl;
    return 0;
}