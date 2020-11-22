#ifndef __INSTANCE_SEG_H__
#define __INSTANCE_SEG_H__

#include "buffers.h"

#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base_model.h"

class InstanceSeg: public BaseModel
{
public:
  InstanceSeg() {}
  InstanceSeg(const YAML::Node& cfg):BaseModel(cfg){}

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs, const YAML::Node& cfg_preprocess);
  float interpolateBilinear(const float* src, int srcH, int srcW, float y, float x);

private:
  uint8_t colors_[19][3] = {
    {244,  67,  54},
    {233,  30,  99},
    {156,  39, 176},
    {103,  58, 183},
    { 63,  81, 181},
    { 33, 150, 243},
    {  3, 169, 244},
    {  0, 188, 212},
    {  0, 150, 136},
    { 76, 175,  80},
    {139, 195,  74},
    {205, 220,  57},
    {255, 235,  59},
    {255, 193,   7},
    {255, 152,   0},
    {255,  87,  34},
    {121,  85,  72},
    {158, 158, 158},
    { 96, 125, 139}
  };
  int num_colors_ = 19;
};

#endif // __INSTANCE_SEG_H__
