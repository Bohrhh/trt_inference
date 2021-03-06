#include "feature_extraction.h"

void FeatureExtraction::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs,
  const YAML::Node& cfg_preprocess,
  cv::Mat& visImg)
{
  visImg = img.clone();
  imgPre::PaddingMode pm = static_cast<imgPre::PaddingMode>(cfg_preprocess["paddingMode"].as<int>());
  if (pm!=imgPre::PaddingMode::NoPadding) {
    throw std::runtime_error("FeatureExtraction model's padding mode should be NoPadding!");
  }

  cv::Mat scores      = outputs["scores"];
  cv::Mat keypoints   = outputs["keypoints"];
  cv::Mat descriptors = outputs["descriptors"];
  int height          = visImg.rows;
  int width           = visImg.cols;
  cv::Scalar color(0,255,0);

  for(int i=0; i<1024; ++i){
    if(scores.at<float>(0,i) < scoreThld_)
      break;
    int x = keypoints.at<float>(0,i,0)*width;
    int y = keypoints.at<float>(0,i,1)*height;
    cv::circle(visImg, cv::Point(x, y), 2, color, -1, 16);
  }
}