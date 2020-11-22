#include "instance_seg.h"

void InstanceSeg::vis(
  cv::Mat& img, 
  std::unordered_map<std::string, cv::Mat>& outputs,
  const YAML::Node& cfg_preprocess)
{
  imgPre::PaddingMode pm = static_cast<imgPre::PaddingMode>(cfg_preprocess["padding_mode"].as<int>());
  if (pm!=imgPre::PaddingMode::NoPadding) {
    throw std::runtime_error("InstanceSeg model's padding mode should be NoPadding!");
  }

  cv::Mat num_detections = outputs["num_detections"];
  cv::Mat nmsed_classes  = outputs["nmsed_classes"];
  cv::Mat nmsed_boxes    = outputs["nmsed_boxes"];
  cv::Mat mask_probs     = outputs["mask_probs"];

  float alpha = 0.6f;
  const int num = int(num_detections.at<float>(0));
  const int maskH = inOutDims_["mask_probs"].d[2];
  const int maskW = inOutDims_["mask_probs"].d[3];
  const int imageH = img.rows;
  const int imageW = img.cols;

  for (int i=num-1; i>=0; i--){
    const float x1 = nmsed_boxes.at<float>(0,i,0)*imageW;
    const float y1 = nmsed_boxes.at<float>(0,i,1)*imageH;
    const float x2 = nmsed_boxes.at<float>(0,i,2)*imageW;
    const float y2 = nmsed_boxes.at<float>(0,i,3)*imageH;
    const float boxH = y2 - y1;
    const float boxW = x2 - x1;
    const float yDelta = maskH / boxH;
    const float xDelta = maskW / boxW;
    const float* src = (float*)mask_probs.data + i*maskH*maskW;
    const int class_type = (int)nmsed_classes.at<float>(0,i);

    for (int y = int(y1); y < int(y2); ++y){
      for (int x = int(x1); x < int(x2); ++x){

        const float ySample = std::max(std::min(yDelta * (y+0.5f-y1), maskH - 0.5f)-0.5f, 0.0f);
        const float xSample = std::max(std::min(xDelta * (x+0.5f-x1), maskW - 0.5f)-0.5f, 0.0f);
        float mask_pixel = interpolateBilinear(src, maskH, maskW, ySample, xSample);
        if (mask_pixel > 0.6){
            float p_r = static_cast<float>(img.at<cv::Vec3b>(y, x)[0]);
            float p_g = static_cast<float>(img.at<cv::Vec3b>(y, x)[1]);
            float p_b = static_cast<float>(img.at<cv::Vec3b>(y, x)[2]);
            img.at<cv::Vec3b>(y, x)[0]
                = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_r * (1 - alpha) + colors_[class_type%num_colors_][0] * alpha)));
            img.at<cv::Vec3b>(y, x)[1]
                = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_g * (1 - alpha) + colors_[class_type%num_colors_][1] * alpha)));
            img.at<cv::Vec3b>(y, x)[2]
                = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_b * (1 - alpha) + colors_[class_type%num_colors_][2] * alpha)));
        }
      }
    }
  }
}

float InstanceSeg::interpolateBilinear(const float* src, int srcH, int srcW, float y, float x){
    const int y0 = static_cast<int>(y);
    const float yAlpha = y - static_cast<float>(y0);
    const int x0 = static_cast<int>(x);
    const float xAlpha = x - static_cast<float>(x0);

    assert(y0 < srcH);
    assert(x0 < srcW);

    const int y1 = (yAlpha == 0) ? y0 : y0 + 1; // ceil
    const int x1 = (xAlpha == 0) ? x0 : x0 + 1; // ceil

    assert(y1 < srcH);
    assert(x1 < srcW);

    const float src00 = src[(y0) *srcW + (x0)];
    const float src01 = src[(y0) *srcW + (x1)];
    const float src10 = src[(y1) *srcW + (x0)];
    const float src11 = src[(y1) *srcW + (x1)];

    const float src0 = src00 * (1 - xAlpha) + src01 * xAlpha;
    const float src1 = src10 * (1 - xAlpha) + src11 * xAlpha;

    return src0 * (1 - yAlpha) + src1 * yAlpha;
}
