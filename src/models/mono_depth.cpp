#include "mono_depth.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

void MonoDepth::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs,
  const YAML::Node& cfg_preprocess,
  cv::Mat& visImg)
{
  imgPre::PaddingMode pm = static_cast<imgPre::PaddingMode>(cfg_preprocess["paddingMode"].as<int>());
  if (pm!=imgPre::PaddingMode::NoPadding) {
    throw std::runtime_error("MonoDepth model's padding mode should be NoPadding!");
  }

  cv::Mat depth = outputs["depth"];

  int height = inOutDims_["depth"].d[2];
  int width  = inOutDims_["depth"].d[3];

  visImg.create(height, width, CV_8UC1);

  for (int y = 0; y < height; y ++)
    for (int x = 0; x < width; x ++)
    {
      int index[4] = {0,0,y,x};
      visImg.at<uchar>(y,x) = static_cast<uchar>(std::min(depth.at<float>(index)/maxDepth_*255, 255.0f));
    }

  cv::applyColorMap(visImg, visImg, cv::COLORMAP_MAGMA);

  if( height!=img.rows || width!=img.cols)
    cv::resize(visImg, visImg, img.size());

  cv::vconcat(img, visImg, visImg);

}
