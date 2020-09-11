#ifndef __STEREO_DEPTH_H__
#define __STEREO_DEPTH_H__

#include "buffers.h"

#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base_model.h"

class StereoDepth: public BaseModel
{
public:
  StereoDepth() {}
  StereoDepth(const YAML::Node& cfg):BaseModel(cfg){
    maxDisp_ = cfg["vis"]["maxDisp"].as<float>();
  }

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs);

private:
  float maxDisp_;

};

#endif // __STEREO_DEPTH_H__
