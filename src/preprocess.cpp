#include "preprocess.h"

Preprocess::Preprocess(const YAML::Node& cfg){
  image_mode_  = cfg["image_mode"].as<int>();
  resize_mode_ = cfg["resize_mode"].as<int>();
  height_      = cfg["height"].as<int>();
  width_       = cfg["width"].as<int>();
  norm_        = static_cast<bool>(cfg["norm"].as<int>());
  padding_     = static_cast<bool>(cfg["padding"].as<int>());
  assert(cfg["mean"].IsSequence());
  for (std::size_t i=0; i<cfg["mean"].size(); ++i) {
    mean_[i] = cfg["mean"][i].as<float>();
  }
  assert(cfg["std"].IsSequence());
  for (std::size_t i=0; i<cfg["std"].size(); ++i) {
    std_[i] = cfg["std"][i].as<float>();
  }
}

void Preprocess::make_preprocess(const cv::Mat& _src, cv::Mat& _dst) {
  cv::Mat src;

  // convert image mode
  if(static_cast<ImageMode>(image_mode_) == ImageMode::RGB)
    cv::cvtColor(_src, src, cv::COLOR_BGR2RGB);
  else if(static_cast<ImageMode>(image_mode_) == ImageMode::GRAY)
    cv::cvtColor(_src, src, cv::COLOR_BGR2GRAY);
  else 
    src = _src;

  // resize
  resize(src, _dst, height_, width_, resize_mode_, padding_);

  // normalize
  normalize(_dst, _dst, mean_, std_, norm_);

}

void Preprocess::normalize(const cv::Mat& _src, cv::Mat& _dst, 
                           const float mean[], const float std[], bool norm) {
  cv::Mat src = _src;
  int height = src.rows;
  int width = src.cols;
  int channels = src.channels();
  float scale = norm ? 255.0f:1.0f;

  assert (channels==1 || channels==3);
  if(channels == 1)
    _dst.create(src.size(), CV_32FC1);
  else if(channels == 3)
    _dst.create(src.size(), CV_32FC3);

  for(int i=0; i<height; ++i)
    for(int j=0; j<width; ++j){
      if(channels==1){
        _dst.at<float>(i, j) = (static_cast<float>(src.at<uchar>(i, j))/scale - mean[0])/std[0];
      }  
      else{
        _dst.at<cv::Vec3f>(i, j)[0] = (static_cast<float>(src.at<cv::Vec3b>(i, j)[0])/scale - mean[0])/std[0];
        _dst.at<cv::Vec3f>(i, j)[1] = (static_cast<float>(src.at<cv::Vec3b>(i, j)[1])/scale - mean[1])/std[1];
        _dst.at<cv::Vec3f>(i, j)[2] = (static_cast<float>(src.at<cv::Vec3b>(i, j)[2])/scale - mean[2])/std[2];
      }
    }
}

void Preprocess::resize(const cv::Mat& _src, cv::Mat& _dst, 
                        const int height, const int width, 
                        int resize_mode, bool padding){
  cv::Mat src = _src;
  int sheight = src.rows;
  int swidth = src.cols;
  int theight = height;
  int twidth = width;

  if(padding){
    if(float(width)/float(swidth)<float(height)/float(sheight)){
      theight = int(float(width)/float(swidth)*sheight);      
    }
    else{
      twidth = int(float(height)/float(sheight)*swidth);
    }
  }

  cv::Mat dst;
  if(static_cast<ResizeMode>(resize_mode) == ResizeMode::Nearest)
    cv::resize(src, dst, cv::Size(twidth, theight), 0.0, 0.0, cv::INTER_NEAREST);
  else if (static_cast<ResizeMode>(resize_mode) == ResizeMode::Bilinear)
    cv::resize(src, dst, cv::Size(twidth, theight), 0.0, 0.0, cv::INTER_LINEAR);

  if(padding){
    _dst = cv::Mat::zeros(cv::Size(width, height), src.type());
    dst.copyTo(_dst(cv::Range(0, theight), cv::Range(0, twidth)));
  }
  else{
    _dst = dst;
  }
}