#ifndef _STEREO_CAMERA_H_
#define _STEREO_CAMERA_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

class StereoCamera
{
public:
  StereoCamera(const YAML::Node& cfg);
  void load(const std::string& filename);
  bool read(cv::Mat& imgL, cv::Mat& imgR);
  bool read(cv::Mat& dst, bool isLeft=true);
  void write(cv::Mat& img);
  void release();

private:
  int height_, width_;
  float rectifyAlpha_;

  cv::Mat cameraMatrixL_, distCoeffsL_;
  cv::Mat cameraMatrixR_, distCoeffsR_;
  cv::Mat R_, T_;
  cv::Mat R1_, R2_, P1_, P2_, Q_;
  cv::Mat mapxL_, mapyL_, mapxR_, mapyR_;

  std::string video_;
  cv::VideoCapture cap_;
  cv::VideoWriter wri_;

  bool remap_;
  bool save_;
  
};

#endif //_STEREO_CAMERA_H_
