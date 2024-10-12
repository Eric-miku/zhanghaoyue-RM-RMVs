#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
class LightDescriptor
{
public:
    LightDescriptor() {};
    LightDescriptor(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
    const LightDescriptor& operator =(const LightDescriptor& ld)
    {
        this->width = ld.width;
        this->length = ld.length;
        this->center = ld.center;
        this->angle = ld.angle;
        this->area = ld.area;
        return *this;
    }
public:
    float width;
    float length;
    cv::Point2f center;
    float angle;
    float area;
};
 
int main(int argc, char** argv)
{
    VideoCapture cap("../images/blue.mp4");
    Mat frame;
    while (true)
    {
        cap.read(frame);
        if (frame.empty())
        {
            break;
        }
        
        vector<Mat>channels;
        split(frame, channels);
        Mat red = channels.at(2);
        inRange(red, Scalar(156, 43, 46), Scalar(180, 255, 255), red);
        Canny(red, red, 0, 30, 3);
        GaussianBlur(red, red, Size(3, 3), 0);
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        dilate(red, red, element);
        morphologyEx(red, red, MORPH_CLOSE, 0);
        vector<vector<Point>>contours;
        vector<Vec4i>hierachy;
       
        findContours(red, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<LightDescriptor> lightInfos;

        for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积
            double area = contourArea(contours[i]);
            // 去除较小轮廓&fitEllipse的限制条件
            if (area < 5 || contours[i].size() <= 1)
                continue;
            // 用椭圆拟合区域得到外接矩形
            RotatedRect Light_Rec = fitEllipse(contours[i]);
 
            // 长宽比和轮廓面积比限制
            if (Light_Rec.size.width / Light_Rec.size.height > 5)
                continue;
            // 扩大灯柱的面积
            Light_Rec.size.height *= 1.1;
            Light_Rec.size.width *= 1.1;
            lightInfos.push_back(LightDescriptor(Light_Rec));
        }
       
        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                LightDescriptor& leftLight = lightInfos[i];
                LightDescriptor& rightLight = lightInfos[j];
 
                //角差
                float angleDiff_ = abs(leftLight.angle - rightLight.angle);
                //长度差比率
                float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                //筛选
                if (angleDiff_ > 10 || LenDiff_ratio > 0.8) {
                    continue;
                }
 
                //左右灯条相距距离
                float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
                //左右灯条长度的平均值
                float meanLen = (leftLight.length + rightLight.length) / 2;
                //左右灯条长度差比值
                float lendiff = abs(leftLight.length - rightLight.length) / meanLen;
                //左右灯条中心点y的差值
                float yDiff = abs(leftLight.center.y - rightLight.center.y);
                //y差比率
                float yDiff_ratio = yDiff / meanLen;
                //左右灯条中心点x的差值
                float xDiff = abs(leftLight.center.x - rightLight.center.x);
                //x差比率
                float xDiff_ratio = xDiff / meanLen;
                //相距距离与灯条长度比值
                float ratio = dis / meanLen;
                //筛选
                if (lendiff > 0.5 ||
                    yDiff_ratio > 1.2 ||
                    xDiff_ratio > 2 ||
                    xDiff_ratio < 0.6 ||
                    ratio > 3.5 ||
                    ratio < 0.5) {
                    continue;
                }
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
                Point2f vertices[4];
                rect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0 ,255), 2);
                }
            }
        }
 
        imshow("装甲板识别1", red);
        imshow("装甲板识别", frame);
        int c = waitKey(30);
        if (c == 27)
            break;
 
 
    }
    waitKey(0);
    return 0;
 
}

