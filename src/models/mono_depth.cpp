#include "mono_depth.h"

void MonoDepth::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs)
{
  cv::Mat depth = outputs["depth"];
  cv::Mat img_color;
  float maxDepth = 20000.0;

  int height = inOutDims_["depth"].d[1];
  int width  = inOutDims_["depth"].d[2];

  img_color.create(height, width, CV_8UC1);

  for (int y = 0; y < height; y ++)
    for (int x = 0; x < width; x ++)
    {
      img_color.at<uchar>(y,x) = static_cast<uchar>(std::min(depth.at<float>(0,y,x)/maxDepth*255, 255.0f));
    }

  cv::applyColorMap(img_color, img_color, cv::COLORMAP_HOT);

  if( height!=img.rows || width!=img.cols)
    cv::resize(img_color, img_color, img.size());

  cv::hconcat(img, img_color, img);

}
