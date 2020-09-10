#ifndef _IMAGE_PREPROCESS_H_
#define _IMAGE_PREPROCESS_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

enum class ImageMode  {RGB, BGR, GRAY};
enum class ResizeMode {Nearest, Bilinear, Hello};

class Preprocess
{
public:
  Preprocess(const YAML::Node& cfg);
  void set_parameters();
  void make_preprocess(const cv::Mat& src, cv::Mat& dst);
  void normalize(const cv::Mat& src, cv::Mat& dst, const float mean[], const float std[], bool norm=true); 
  void resize(const cv::Mat& src, cv::Mat& dst, const int height, const int width, int resize_mode, bool padding=false);

private:
  int image_mode_;
  int resize_mode_;
  int height_;
  int width_;
  bool norm_;
  bool padding_;
  float mean_[3] = {0.0f, 0.0f, 0.0f};
  float std_[3]  = {1.0f, 1.0f, 1.0f};
};


#endif //_IMAGE_PREPROCESS_H_
