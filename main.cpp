#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <tuple>
#include <glob.h>

#include "FaceDetector.h"

using namespace std;

int main(int argc, char **argv)
{
    string imgPath;
    if (argc = 1)
    {
        // imgPath = "../unit_test/unit_test_mask.png";
        imgPath = "../unit_test/leo.jpg";
    }
    else if (argc = 2)
    {
        imgPath = argv[1];
    }
    // cv::VideoCapture cap(0);
    string param = "../model/mobilenet0.25_201205_320_sunglasses_Final-sim.param";
    string bin = "../model/mobilenet0.25_201205_320_sunglasses_Final-sim.bin";
    const int max_side = 320;

    // retinaface
    Detector detector(param, bin, true);
    Timer timer;
    for (int i = 0; i < 10; i++)
    {
        // while(cv::waitKey(1) != 27){
        cv::Mat img_ = cv::imread(imgPath.c_str());
        // cv::Mat img_;
        // cap >> img_;

        cv::Mat src = img_.clone();
        cv::Mat img;
        img_.convertTo(img, CV_32FC3);
        // std::cout << img.size() << std::endl;
        // scale
        float long_side = std::max(src.cols, src.rows);
        float scale = max_side / long_side;
        cv::Mat img_scale;
        cv::resize(src, img_scale, cv::Size(src.cols * scale, src.rows * scale));
        // cout << img_scale.size()<< endl;
        // cv::imshow("as", src);
        std::vector<FaceCrop> boxes;

        timer.tic();

        detector.Detect(img_scale, boxes);
        timer.toc("----total timer:");

        // std::cout << boxes.size() << std::endl;
        // draw image
        for (int j = 0; j < boxes.size(); ++j)
        {
            cv::Rect rect(boxes[j].x1 / scale, boxes[j].y1 / scale, boxes[j].x2 / scale - boxes[j].x1 / scale, boxes[j].y2 / scale - boxes[j].y1 / scale);
            char test[80];
            sprintf(test, "%.4f", boxes[j].face_score);
            char mask[80];
            sprintf(mask, "%.4f", boxes[j].mask_score);
            float mask_threshold = 0;
            if (boxes[j].mask_score > mask_threshold)
            {
                cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            }
            else
            {
                cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 1, 8, 0);
            }
            cv::putText(img, test, cv::Size((boxes[j].x1 / scale), boxes[j].y1 / scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::putText(img, mask, cv::Size((boxes[j].x2 / scale), boxes[j].y2 / scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
            cv::circle(img, cv::Point(boxes[j].lmks[0] / scale, boxes[j].lmks[1] / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].lmks[2] / scale, boxes[j].lmks[3] / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].lmks[4] / scale, boxes[j].lmks[5] / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].lmks[6] / scale, boxes[j].lmks[7] / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].lmks[8] / scale, boxes[j].lmks[9] / scale), 1, cv::Scalar(255, 0, 0), 4);
        }

        cv::imwrite("../test.png", img);
    }
    return 0;
}
