/*
成功识别版本，灰度识别法
开始识别主板
加入灯条类
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
    while (true) {
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
    //countings_finding
    
    cv::GaussianBlur(binaryImage, binaryImage, cv::Size(5, 5), 0); // 平滑处理
    cv::Mat edges;

    cv::Canny(binaryImage, edges, 50, 150); // 使用 Canny 边缘检测

    vector<vector<Point>> contours;
    vector<LightDescriptor> lightInfos;
    findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        
       RotatedRect rotatedRect = cv::minAreaRect(contour); // 获取最小外接矩形
        // 计算长宽比
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 过滤条件
        if ( aspectRatio > 4.0 && rotatedRect.size.area() > 100) {
            // 绘制旋转矩形
            lightInfos.push_back(LightDescriptor(rotatedRect));
            
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2); // 绿色框
            }
            
        }
    }
    for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; j < lightInfos.size(); j++) {
                LightDescriptor& leftLight = lightInfos[i];
                LightDescriptor& rightLight = lightInfos[j];
                
                //角差
                float angleDiff_ = abs(leftLight.angle - rightLight.angle);
                //长度差比率
                float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                //筛选
                if (angleDiff_ > 15  || LenDiff_ratio > 0.8) {
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

                /*
                //筛选
                
                     if (lendiff > 0.5 ||
                    yDiff_ratio > 1.2 ||
                    xDiff_ratio > 2 ||
                    xDiff_ratio < 0.6 ||
                    ratio > 3.5 ||
                    ratio < 0.5) {
                    continue;
                }*/
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
                Point2f vertices[4];
                rect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0 ,255), 2);
                }
            }
        }
    imshow("Edges", edges);
    imshow("binaryImage", binaryImage);
    imshow("image", image);
}

