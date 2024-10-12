//初版识别
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
void image_process_gray(Mat&);
void image_process_hsv(Mat&);
void image_process_RBG(Mat& );
int main() {
    // 打开视频文件
    //cv::VideoCapture cap("task/images/blue.mp4");
    cv::VideoCapture cap("../images/red.mp4");

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
    /*
    cv::GaussianBlur(binaryImage, binaryImage, cv::Size(5, 5), 0); // 平滑处理
    cv::Mat edges;

    cv::Canny(binaryImage, edges, 50, 150); // 使用 Canny 边缘检测

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
       cv::RotatedRect rotatedRect = cv::minAreaRect(contour); // 获取最小外接矩形
        
        // 计算长宽比
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 过滤条件
        if (aspectRatio > 2.0 && aspectRatio < 5.0 && rotatedRect.size.area() > 100) {
            // 绘制旋转矩形
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(binaryImage, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2); // 绿色框
            }
        }
    }*/

    // 3.1 自适应阈值
    cv::Mat adaptiveBinaryImage;
    cv::adaptiveThreshold(grayImage, adaptiveBinaryImage, 255, 
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY, 11, 2);
    //countings_finding
    cv::GaussianBlur(adaptiveBinaryImage, adaptiveBinaryImage, cv::Size(5, 5), 0); // 平滑处理
    cv::Mat edges;

    cv::Canny(adaptiveBinaryImage, edges, 50, 150); // 使用 Canny 边缘检测

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
       cv::RotatedRect rotatedRect = cv::minAreaRect(contour); // 获取最小外接矩形
        
        // 计算长宽比
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 过滤条件
        if (aspectRatio > 2.0 && aspectRatio < 5.0 && rotatedRect.size.area() > 100) {
            // 绘制旋转矩形
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(adaptiveBinaryImage, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2); // 绿色框
            }
        }
    }

    
    
    // 7. 显示结果

    //cv::imshow("Edges", edges);



    //cv::imshow("Original Image", image);
    //cv::imshow("Gray Image", grayImage);
    //cv::imshow("Binary Image (Fixed Threshold)", binaryImage);
    cv::imshow("Binary Image (Adaptive Threshold)", adaptiveBinaryImage);
}
void image_process_hsv(Mat& image){
    
    // 2. 转换颜色空间
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    cv::Mat mask;

    cv::Scalar lowerRed1(35, 43, 46); // H 范围 10-20
    cv::Scalar upperRed1(99, 255, 255);
    cv::inRange(hsvImage, lowerRed1, upperRed1, mask); // 创建二值化
    cv::imshow("hsvImage", hsvImage);
    // 4. 形态学操作
    cv::Mat morphMask;
    cv::morphologyEx(mask, morphMask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2); // 闭运算填补空洞
    cv::morphologyEx(morphMask, morphMask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2); // 开运算去除噪声

    // 5. 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morphMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 6. 过滤轮廓
    for (const auto& contour : contours) {
        cv::RotatedRect rotatedRect = cv::minAreaRect(contour); // 获取最小外接矩形
        
        // 计算长宽比
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 过滤条件
        if (aspectRatio > 2.0 && aspectRatio < 6.0 && rotatedRect.size.area() > 100&& rotatedRect.size.area() < 1000) {
            // 绘制旋转矩形
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2); // 绿色框
            }
        }
    }

    // 7. 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Mask", morphMask);
}
void image_process_RBG(Mat& image){
    // 2. 转换颜色空间为 RGB
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB); // OpenCV 默认使用 BGR，需转换为 RGB

    imshow("rbgImage", rgbImage);
    // 3. 创建二值化掩码
    cv::Mat mask;
    // 偏黄红色的 RGB 阈值范围
    cv::Scalar lowerBlue(0, 100, 100);   // R > 100, G < 50, B < 50
    cv::Scalar upperBlue(100, 200, 255); // R < 255, G < 150, B < 150
    cv::inRange(rgbImage, lowerBlue, upperBlue, mask); // 创建二值化图像

    // 4. 形态学操作
    cv::Mat morphMask;
    cv::morphologyEx(mask, morphMask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2); // 闭运算填补空洞
    cv::morphologyEx(morphMask, morphMask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2); // 开运算去除噪声

    // 5. 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morphMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 6. 过滤轮廓
    for (const auto& contour : contours) {
        cv::RotatedRect rotatedRect = cv::minAreaRect(contour); // 获取最小外接矩形
        
        // 计算长宽比
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 过滤条件
        if (aspectRatio > 2.0 && aspectRatio < 5.0 && rotatedRect.size.area() > 100) {
            // 绘制旋转矩形
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2); // 绿色框
            }
        }
    }

    // 7. 显示结果
    cv::imshow("Original Image", image);
    cv::imshow("Mask", morphMask);
}