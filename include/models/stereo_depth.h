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

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs, const YAML::Node& cfg_preprocess);

private:
  float maxDisp_=40;
  float colorMap_[8][4] = {{0,0,0,114},{0,0,1,185},{1,0,0,114},{1,0,1,174},
                           {0,1,0,114},{0,1,1,185},{1,1,0,114},{1,1,1,0  }};


};

#endif // __STEREO_DEPTH_H__
