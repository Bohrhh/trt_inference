#ifndef _IMAGE_PREPROCESS_H_
#define _IMAGE_PREPROCESS_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>


namespace imgPre{
  enum class ImageMode  {RGB, BGR, GRAY};
  enum class ResizeMode {Nearest, Bilinear};
  enum class PaddingMode {NoPadding, LeftTop, LetterBox};
}


class Preprocess
{
public:
  Preprocess(const YAML::Node& cfg);

/**
 * @brief main workflow, do image preprocess
 * @param[in]  src
 * @param[out] dst
 */
  void makePreprocess(const cv::Mat& src, cv::Mat& dst);

  void resize(const cv::Mat& src, cv::Mat& dst, const int height, const int width, 
              imgPre::ResizeMode resizeMode, imgPre::PaddingMode padding, uchar paddingValue=0);
              
  void normalize(const cv::Mat& src, cv::Mat& dst, const float mean[], const float std[], bool norm=true); 

private:
  int imageMode_;
  int resizeMode_;
  int height_;
  int width_;
  bool norm_;
  int paddingMode_;
  uchar paddingValue_ = 0;
  float mean_[3] = {0.0f, 0.0f, 0.0f};
  float std_[3]  = {1.0f, 1.0f, 1.0f};
};


#endif //_IMAGE_PREPROCESS_H_
