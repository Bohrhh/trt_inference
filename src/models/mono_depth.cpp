#include "mono_depth.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

void MonoDepth::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs)
{
  cv::Mat depth = outputs["depth"];
  cv::Mat img_color;

  int height = inOutDims_["depth"].d[2];
  int width  = inOutDims_["depth"].d[3];

  img_color.create(height, width, CV_8UC1);

  for (int y = 0; y < height; y ++)
    for (int x = 0; x < width; x ++)
    {
      int index[4] = {0,0,y,x};
      img_color.at<uchar>(y,x) = static_cast<uchar>(std::min(depth.at<float>(index)/maxDepth_*255, 255.0f));
    }

  cv::applyColorMap(img_color, img_color, cv::COLORMAP_HOT);

  if( height!=img.rows || width!=img.cols)
    cv::resize(img_color, img_color, img.size());

  cv::hconcat(img, img_color, img);

}
