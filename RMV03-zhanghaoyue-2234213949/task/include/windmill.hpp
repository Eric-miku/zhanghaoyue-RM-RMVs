#ifndef WINDMILL_H_
#define WINDMILL_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <random>

namespace WINDMILL
{
    //风车类
    class WindMill
    {
    private:
        int cnt;
        bool direct;
        double A;
        double w;
        double A0;
        double fai;
        double now_angle;
        double start_time;
        cv::Point2i R_center;
        //R仅考虑所在位置
        void drawR(cv::Mat &img, const cv::Point2i &center);
        //两种扇叶要考虑角度
        void drawHitFan(cv::Mat &img, const cv::Point2i &center, double angle);
        void drawOtherFan(cv::Mat &img, const cv::Point2i &center, double angle);
        //
        cv::Point calPoint(const cv::Point2f &center, double angle_deg, double r)
        {
            return center + cv::Point2f((float)cos(angle_deg / 180 * 3.1415926), (float)-sin(angle_deg / 180 * 3.1415926)) * r;
        }
        double SumAngle(double angle_now, double t0, double dt)//t0与dt都是时间
        {
            //反映了弧度随时间的变化
            double dangle = A0 * dt + (A / w) * (cos(w * t0 + 1.81) - cos(w * (t0 + dt) + 1.81));//这里的A0是b
            angle_now += dangle / 3.1415926 * 180;
            if (angle_now < 0)
            {
                angle_now = 360 + angle_now;
            }
            if (angle_now > 360)
            {
                angle_now -= 360;
            }
            return angle_now;
        }

    public:
        WindMill(double time = 0);
        cv::Mat getMat(double time);

    };
} // namespace WINDMILL

#endif