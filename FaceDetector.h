#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include "net.h"
#include <chrono>
using namespace std::chrono;

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
                printf("%f\n", 1000/diff);
        }

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

struct boubox{
    float cx;
    float cy;
    float sx;
    float sy;
};
struct FaceCrop {
    float x1;
    float y1;
    float x2;
    float y2;
    float area;
    float regreCoord[4];
    std::vector<float> lmks;
    float face_score = -1;
    float real_score = 1;
    float leye_open = -1;
    float reye_open = -1;
    float pose_x;
    float pose_y;
    float pose_z;
    float blur_score = 0;
    float brightness_score;
    float darkness_score;
    float out_of_range;
    float mask_score = 0;
    int leye_x1;
    int leye_y1;
    int leye_x2;
    int leye_y2;

    int reye_x1;
    int reye_y1;
    int reye_x2;
    int reye_y2;
};
class Detector
{

public:
    Detector();

    ~Detector();

    Detector(const std::string &model_param, const std::string &model_bin, bool retinaface = true);

    void loadModel(const std::string &model_param, const std::string &model_bin);
    
    void releaseModels ();

    // inline void Release();

    void nms(std::vector<FaceCrop> &input_boxes, float NMS_THRESH);

    void Detect(cv::Mat& bgr, std::vector<FaceCrop>& boxes);

    void create_anchor_retinaface(std::vector<boubox> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(FaceCrop a, FaceCrop b);

public:
    float _nms = 0.4;
    float _threshold = 0.6;
    float _mean_val[3] = {104.f, 117.f, 123.f};
    bool _retinaface = true;

    ncnn::Net *Net;
};
#endif //
