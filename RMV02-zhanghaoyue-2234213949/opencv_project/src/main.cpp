#include <iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void grey_image(Mat);
void hsv_image(Mat);

void j_image(Mat);
void g_image(Mat);

void Red_image(Mat);

void draw_image(Mat);

void process_image(Mat);
int main(){
    cout<<"1"<<endl;
    //    namedWindow("image",WINDOW_AUTOSIZE);
    Mat image = imread("../resources/test_image.png");
//图像颜色空间转换
    grey_image(image);
    hsv_image(image);
//应用各种滤波操作
    j_image(image);
    g_image(image);
//特征提取
    Red_image(image);
//图像绘制
    draw_image(image);
//对图像进行处理 
    // 图像旋转 35 度
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, 35, 1.0);
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size());

    // 图像裁剪为左上角 1/4
    Mat croppedImage = image(Rect(0, 0, image.cols / 2, image.rows / 2));

    // 储存结果
    //imshow("../resources/5.1Rotated_Image", rotatedImage);
    imshow("../resources/5.2Cropped_Image", croppedImage);
    waitKey(0);
    return 0;
}


//图像颜色空间转换——转化为灰度图
void grey_image(Mat image){
    Mat grey_image; 
    cvtColor(image, grey_image,COLOR_BGR2GRAY);
    imwrite("../resources/1.1grey_image.png", grey_image);
}

//图像颜色空间转换——转化为HSV图
void hsv_image(Mat image){
    Mat hsv_image; 
    cvtColor(image, hsv_image,COLOR_BGR2HSV);
    imwrite("../resources/1.2hsv_image.png", hsv_image);
}

//应用各种滤波操作——应用均值滤波
void j_image(Mat image){
    Mat j_image;
    blur( image, j_image, Size(15,15),Point(-1,-1));
    imwrite("../resources/2.1j_image.png", j_image);
}

//应用各种滤波操作——应用高斯滤波
void g_image(Mat image){
    Mat g_image;
    GaussianBlur( image, g_image, Size(15,15),0,0);
    imwrite("../resources/2.1g_image.png", g_image);
}

//特征提取——提取红色颜色区域——HSV 方法 
void Red_image(Mat image){
    Mat red_image;
    cvtColor(image, red_image, COLOR_BGR2HSV); // 转为HSV图片
    // 定义红色的HSV范围
    Scalar lower_red1(0, 100, 100); // 较低的红色范围
    Scalar upper_red1(10, 255, 255);
    Scalar lower_red2(160, 100, 100); // 较高的红色范围
    Scalar upper_red2(180, 255, 255);
    // 创建掩膜
    Mat mask1, mask2, mask;
    inRange(red_image, lower_red1, upper_red1, mask1);
    inRange(red_image, lower_red2, upper_red2, mask2);
    // 合并
    bitwise_or(mask1, mask2, mask); 
    red_image=mask;
    //显示
    //namedWindow("red_image", WINDOW_NORMAL);
    //resizeWindow("red_image", 800, 800);
    //imshow("red_image", red_image);
    waitKey(0);
    imwrite("../resources/3.1red_image.png", red_image);
    // 寻找红色的外轮廓
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 寻找 bounding box 和计算面积
    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour);
        double area = contourArea(contour);

        // 绘制红色外轮廓和 bounding box
        rectangle(mask, boundingBox, Scalar(0, 255, 0), 2);
        drawContours(mask, contours, -1, Scalar(255, 0, 0), 2);   
        cout << "Area: " << area << endl;
    }
    imwrite("../resources/3.2red_bounding.png", mask);

// 高亮颜色区域处理
    //灰度化和二值化
    Mat grayImage, binaryImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    threshold(grayImage, binaryImage, 128, 255, THRESH_BINARY);
    imwrite("../resources/3.3grayImage.png", grayImage);
    imwrite("../resources/3.4binaryImage.png", binaryImage);

    // 膨胀和腐蚀
    Mat dilatedImage, erodedImage;
    dilate(binaryImage, dilatedImage, Mat());
    erode(dilatedImage, erodedImage, Mat());
    imwrite("../resources/3.5dilatedImage.png", dilatedImage);
    imwrite("../resources/3.6erodedImage.png", erodedImage);
    // 漫水处理
    Mat watershedImage = erodedImage.clone();
    watershedImage.setTo(Scalar(0));
    imwrite("../resources/3.7watershedImage.png", watershedImage);
}

//图像绘制 
void draw_image(Mat image){
    // 绘制圆形、方形和文字
    circle(image, Point(100, 100), 50, Scalar(255, 255, 0), 2); // 圆形
    rectangle(image, Point(200, 200), Point(300, 300), Scalar(255, 0, 255), 2); // 方形
    putText(image, "Hello", Point(400, 400), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2); // 文字
    imwrite("../resources/4.1_image.png", image);

}

//对图像进行处理 
void process_image(Mat image){
    // 图像旋转 35 度
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, 35, 1.0);
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size());

    // 图像裁剪为左上角 1/4
    Mat croppedImage = image(Rect(0, 0, image.cols / 2, image.rows / 2));

    // 储存结果
    //imwrite("../resources/5.1Rotated_Image", rotatedImage);
    //imwrite("../resources/5.2Cropped_Image", croppedImage);
}