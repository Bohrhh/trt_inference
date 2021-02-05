#include "stereo_depth.h"

void StereoDepth::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs,
  const YAML::Node& cfg_preprocess)
{
  imgPre::PaddingMode pm = static_cast<imgPre::PaddingMode>(cfg_preprocess["padding_mode"].as<int>());
  if (pm!=imgPre::PaddingMode::NoPadding) {
    throw std::runtime_error("StereoDepth model's padding mode should be NoPadding!");
  }

  cv::Mat disp = outputs["disp"];
  cv::Mat disp_vis;
  int height = inOutDims_["disp"].d[2];
  int width  = inOutDims_["disp"].d[3];
  disp_vis.create(height, width, CV_8UC3);

  float sum = 0;
  for (int32_t i=0; i<8; i++)
    sum += colorMap_[i][3];

  float weights[8]; // relative weights
  float cumsum[8];  // cumulative weights
  cumsum[0] = 0;
  for (int32_t i=0; i<7; i++) {
    weights[i]  = sum/colorMap_[i][3];
    cumsum[i+1] = cumsum[i] + colorMap_[i][3]/sum;
  }

  // for all pixels do
  for (int32_t v=0; v<height; v++) {
    for (int32_t u=0; u<width; u++) {
    
      // get normalized value
      int index[4] = {0,0,v,u};
      float val = std::min(std::max(disp.at<float>(index)/maxDisp_,0.0f),1.0f);
      
      // find bin
      int32_t i;
      for (i=0; i<7; i++)
        if (val<cumsum[i+1])
          break;

      // compute red/green/blue values
      float   w = 1.0-(val-cumsum[i])*weights[i];
      uint8_t r = (uint8_t)((w*colorMap_[i][0]+(1.0-w)*colorMap_[i+1][0]) * 255.0);
      uint8_t g = (uint8_t)((w*colorMap_[i][1]+(1.0-w)*colorMap_[i+1][1]) * 255.0);
      uint8_t b = (uint8_t)((w*colorMap_[i][2]+(1.0-w)*colorMap_[i+1][2]) * 255.0);
      
      // set pixel
      disp_vis.at<cv::Vec3b>(v,u) = cv::Vec3b(b,g,r);
    }
  }

  if( height!=img.rows || width!=img.cols)
    cv::resize(disp_vis, disp_vis, img.size());
  
  cv::vconcat(img, disp_vis, img);
}