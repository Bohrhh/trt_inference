#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include "buffers.h"
#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base_model.h"

namespace Yolo{
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[4];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

class YOLOV5: public BaseModel
{
public:
  YOLOV5() {}
  YOLOV5(const YAML::Node& cfg):BaseModel(cfg){
    confThld_ = cfg["confThld"].as<float>();
    nmsThld_ = cfg["nmsThld"].as<float>();
  }

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs, const YAML::Node& cfg_preprocess);
  float inference_output(cv::Mat& x, float* &start, float confThld, int anchors, int h, int w, int classes, float stride, int det);

private:
  float anchor_grids_[3][6] = {
    {10,13, 16,30, 33,23},
    {30,61, 62,45, 59,119},
    {116,90, 156,198, 373,326}
  };

  float confThld_ = 0.25;
  float nmsThld_ = 0.6;

};


#endif // __YOLOV5_H__
