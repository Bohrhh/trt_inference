#include "stereo_depth.h"

void StereoDepth::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs)
{
  cv::Mat disp = outputs["disp"];
  cv::Mat hsv;
  float maxDisp = 60.0;

  int height = inOutDims_["disp"].d[2];
  int width  = inOutDims_["disp"].d[3];

  hsv.create(height, width, CV_8UC3);

  for (int y = 0; y < height; y ++)
    for (int x = 0; x < width; x ++)
    {
      int index[4] = {0,0,y,x};
      float v = 1 - std::min(disp.at<float>(index), maxDisp) / maxDisp;
      hsv.at<cv::Vec3b>(y,x) = cv::Vec3b((unsigned char)(150 * v), 255, 255);
    }

  cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

  if( height!=img.rows || width!=img.cols)
    cv::resize(hsv, hsv, img.size());

  cv::hconcat(img, hsv, img);

}
