#include "yolov5.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = 0.6) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0]; i++) {
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

cv::Rect get_rect(cv::Mat& img, float bbox[4], int input_h, int input_w) {
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (input_h - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (input_h - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (input_w - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (input_w - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

void YOLOV5::vis(
  cv::Mat& img,
  std::unordered_map<std::string, cv::Mat>& outputs,
  const YAML::Node& cfg_preprocess)
{
  cv::Mat out8  = outputs["output8"];    // 1,3,80,80,85
  cv::Mat out16 = outputs["output16"];   // 1,3,40,40,85
  cv::Mat out32 = outputs["output32"];   // 1,3,20,20,85

  imgPre::PaddingMode pm = static_cast<imgPre::PaddingMode>(cfg_preprocess["padding_mode"].as<int>());
  if (pm!=imgPre::PaddingMode::LetterBox) {
    throw std::runtime_error("Yolov5 model's padding mode should be LetterBox!");
  }

  size_t data_len = (20*20*3+40*40*3+80*80*3)*6;
  float* output = new float[data_len+1]; // (n, 6)
  *output = 0;
  float num_det = 0;

  float* start = output+1;
  num_det = inference_output(out8, start, conf_thresh_, 3, 80, 80, 80, 8, 0);
  *output = *output + num_det;
  num_det = inference_output(out16, start, conf_thresh_, 3, 40, 40, 80, 16, 1);
  *output = *output + num_det;
  num_det = inference_output(out32, start, conf_thresh_, 3, 20, 20, 80, 32, 2);
  *output = *output + num_det;

  std::vector<Yolo::Detection> res;
  nms(res, output, nms_thresh_);
  
  for (size_t j = 0; j < res.size(); j++) {
      cv::Rect r = get_rect(img, res[j].bbox, cfg_preprocess["height"].as<int>(), cfg_preprocess["width"].as<int>());
      cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
  }


  delete[] output;

}

float YOLOV5::inference_output(cv::Mat& x, float* &start, float conf_thresh, int anchors, int h, int w, int classes, float stride, int det){
  // float* data = (float*) x.data;
  // for (int i=0; i<anchors*h*w*(classes+5); ++i){
  //   *data = 1.0f/(1.0f+expf(*data));
  // }
  float* data = (float*) x.data;
  float num_det = 0;
  int xstrides[5] = {85*3*w*h,85*w*h,85*w,85,1};
  for (int i=0; i<anchors; ++i)
    for (int j=0; j<h; ++j)
      for (int k=0; k<w; ++k){
        int ijk = xstrides[1]*i+xstrides[2]*j+xstrides[3]*k;
        float box_score = data[ijk+4];
        if (box_score<conf_thresh) continue;
        float max_class_probs = -1;
        float class_id = 0;
        for (int l=5; l<classes+5; ++l){
          if (data[ijk+l]>max_class_probs){
            max_class_probs = data[ijk+l];
            class_id = l-5;
          }
        }
        if (max_class_probs*box_score<conf_thresh) continue;

        num_det += 1;
        *start = (data[ijk]*2.0-0.5+k)*stride;
        *(start+1) = (data[ijk+1]*2.0-0.5+j)*stride;
        *(start+2) = data[ijk+2]*2.0*data[ijk+2]*2.0*anchor_grids_[det][i*2];
        *(start+3) = data[ijk+3]*2.0*data[ijk+3]*2.0*anchor_grids_[det][i*2+1];
        *(start+4) = max_class_probs*box_score;
        *(start+5) = class_id;
        start += 6;
      }
  return num_det;
}