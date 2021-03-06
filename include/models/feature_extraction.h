#ifndef __FEATURE_EXTRACTION_H__
#define __FEATURE_EXTRACTION_H__

#include "buffers.h"

#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base_model.h"

class FeatureExtraction: public BaseModel
{
public:
  FeatureExtraction() {}
  FeatureExtraction(const YAML::Node& cfg):BaseModel(cfg){
    scoreThld_ = cfg["vis"]["scoreThld"].as<float>();
  }

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs, const YAML::Node& cfgPreprocess, cv::Mat& visImg);

private:
  float scoreThld_=0.01;

};

#endif // __FEATURE_EXTRACTION_H__
