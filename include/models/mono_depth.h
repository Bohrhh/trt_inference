#ifndef __MONO_DEPTH_H__
#define __MONO_DEPTH_H__

#include "buffers.h"

#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base_model.h"

class MonoDepth: public BaseModel
{
public:
  MonoDepth() {}
  MonoDepth(const YAML::Node& cfg):BaseModel(cfg){
    maxDepth_ = cfg["vis"]["maxDepth"].as<float>();
  }

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs, const YAML::Node& cfgPreprocess, cv::Mat& visImg);

private:
  float maxDepth_;

};

#endif // __MONO_DEPTH_H__
