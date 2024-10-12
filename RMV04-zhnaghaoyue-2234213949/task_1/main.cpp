/*
从矩形到椭圆
基本完成识别功能，判断部分为扒来的
*/

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
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

void image_process_gray(Mat&);

int main() {
    // 打开视频文件
    //cv::VideoCapture cap("task/images/blue.mp4");
    cv::VideoCapture cap("../images/blue.mp4");

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file!" << std::endl;
        return -1;
    }

    // 循环读取视频帧
    cv::Mat frame;
    while (waitKey(0)) {
        // 从视频中读取一帧
        bool isSuccess = cap.read(frame);

        // 如果无法读取，退出循环
        if (!isSuccess) {
            std::cout << "Video has ended or cannot read the frame" << std::endl;
            break;
        }
        image_process_gray(frame);
        // 在窗口中显示当前帧
        //cv::imshow("Video Frame", frame);

        // 等待按键，如果按下 'q' 键退出
        if (cv::waitKey(100) == 'q') {
            break;
        }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void image_process_gray(Mat& image){
     // 2. 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 3. 二值化处理
    // 使用固定阈值
    cv::Mat binaryImage;
    double thresholdValue = 190; // 阈值，根据实际情况调整
    cv::threshold(grayImage, binaryImage, thresholdValue, 255, cv::THRESH_BINARY);
    
    cv::GaussianBlur(binaryImage, binaryImage, cv::Size(5, 5), 0); // 平滑处理
    cv::Mat edges;

    Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    dilate(binaryImage, binaryImage, element);  // 膨胀操作

    cv::Canny(binaryImage, edges, 50, 150); // 使用 Canny 边缘检测

    vector<vector<Point>> contours;
    vector<LightDescriptor> lightInfos;
    findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (contour.size() >= 5) {
            cv::RotatedRect ellipse = cv::minAreaRect(contour);
            cv::ellipse(image, ellipse, cv::Scalar(255, 0, 0), 2);
        } else {
            std::cout << "Contour too small, skipping" << std::endl;
        }
    }

    for (const auto& contour : contours) {
        
       RotatedRect rotatedRect = fitEllipse(contour); // 获取最小外接矩形
        // 计算长宽比
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 过滤条件
        if ( aspectRatio > 3.0 && rotatedRect.size.area() > 100) {
            // 绘制旋转矩形
            lightInfos.push_back(LightDescriptor(rotatedRect));
            
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2); // 绿色框
            }
            
        }
    }
    //获得两个面积最大灯条
    for (size_t i = 0; i < lightInfos.size(); i++) {
        for (size_t j = i + 1; j < lightInfos.size(); j++) {
            LightDescriptor& leftLight = lightInfos[i];
            LightDescriptor& rightLight = lightInfos[j];
            if(leftLight.area<rightLight.area){
                LightDescriptor temp=leftLight;
                leftLight=rightLight;
                rightLight=temp;
            }
        }
    }
    LightDescriptor& leftLight = lightInfos[0];
    LightDescriptor& rightLight = lightInfos[1];
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
    if(abs(leftLight.angle - rightLight.angle)>90){
        rightLight.angle+=180;
    }
    Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
    RotatedRect rect = RotatedRect(center, Size(dis, 2*meanLen), (leftLight.angle + rightLight.angle) / 2);
    cout<<(leftLight.angle + rightLight.angle) / 2<<"   "<<leftLight.angle<<"   "<<rightLight.angle<<endl;
    Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++) {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0 ,255), 2);
    }
    imshow("Edges", edges);
    imshow("binaryImage", binaryImage);
    imshow("image", image);
}

