#include "feature_extraction.h"

void FeatureExtraction::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs)
{
  cv::Mat scores      = outputs["scores"];
  cv::Mat keypoints   = outputs["keypoints"];
  cv::Mat descriptors = outputs["descriptors"];
  int height          = img.rows;
  int width           = img.cols;
  cv::Scalar color(255,0,0);

  for(int i=0; i<1024; ++i){
    if(scores.at<float>(0,i) <= 0.5)
      break;
    int x = keypoints.at<float>(0,i,0)*width;
    int y = keypoints.at<float>(0,i,1)*height;
    cv::circle(img, cv::Point(x, y), 1, color, -1, 16);
  }
}