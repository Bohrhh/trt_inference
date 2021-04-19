#include "preprocess.h"


Preprocess::Preprocess(const YAML::Node& cfg){
  image_mode_    = cfg["image_mode"].as<int>();
  resize_mode_   = cfg["resize_mode"].as<int>();
  height_        = cfg["height"].as<int>();
  width_         = cfg["width"].as<int>();
  norm_          = static_cast<bool>(cfg["norm"].as<int>());
  padding_mode_  = cfg["padding_mode"].as<int>();
  padding_value_ = cfg["padding_value"].as<uchar>();
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
  if(static_cast<imgPre::ImageMode>(image_mode_) == imgPre::ImageMode::RGB)
    cv::cvtColor(_src, src, cv::COLOR_BGR2RGB);
  else if(static_cast<imgPre::ImageMode>(image_mode_) == imgPre::ImageMode::GRAY)
    cv::cvtColor(_src, src, cv::COLOR_BGR2GRAY);
  else 
    src = _src;

  // resize
  resize(src, _dst, height_, width_, static_cast<imgPre::ResizeMode>(resize_mode_), 
         static_cast<imgPre::PaddingMode>(padding_mode_), padding_value_);

  // normalize
  normalize(_dst, _dst, mean_, std_, norm_);
}


void Preprocess::resize(const cv::Mat& _src, cv::Mat& _dst, 
                        const int height, const int width, 
                        imgPre::ResizeMode resize_mode, imgPre::PaddingMode padding_mode, uchar padding_value){
  cv::Mat src = _src;
  int sheight = src.rows;
  int swidth = src.cols;
  int theight = height;
  int twidth = width;
  
  if(padding_mode!=imgPre::PaddingMode::NoPadding){
    if(float(width)/float(swidth)<float(height)/float(sheight)){
      theight = int(float(width)/float(swidth)*sheight);      
    }else{
      twidth = int(float(height)/float(sheight)*swidth);
    }
  }

  cv::Mat dst;
  if(resize_mode == imgPre::ResizeMode::Nearest)
    cv::resize(src, dst, cv::Size(twidth, theight), 0.0, 0.0, cv::INTER_NEAREST);
  else if (resize_mode == imgPre::ResizeMode::Bilinear)
    cv::resize(src, dst, cv::Size(twidth, theight), 0.0, 0.0, cv::INTER_LINEAR);

  assert (src.type()==CV_8UC3 || src.type()==CV_8UC1);
  if(src.type()==CV_8UC3)
    _dst = cv::Mat(height, width, src.type(), cv::Scalar(padding_value, padding_value, padding_value));
  else if(src.type()==CV_8UC1)
    _dst = cv::Mat(height, width, src.type(), cv::Scalar(padding_value));

  if(padding_mode==imgPre::PaddingMode::LetterBox){
    int y = (height-theight)/2;
    int x = (width-twidth)/2;
    dst.copyTo(_dst(cv::Range(y, y+theight), cv::Range(x, x+twidth)));
  }else if(padding_mode==imgPre::PaddingMode::LeftTop){
    dst.copyTo(_dst(cv::Range(0, theight), cv::Range(0, twidth)));
  }else {
    _dst = dst;
  }
}


void Preprocess::normalize(const cv::Mat& _src, cv::Mat& _dst, 
                           const float mean[], const float std[], bool norm) {
  cv::Mat src = _src;
  int height = src.rows;
  int width = src.cols;
  int channels = src.channels();
  float scale = norm ? 255.0f:1.0f;

  if(channels!=1 && channels!=3){
    std::cerr << "Invalid input channels" << std::endl;
    exit(1);
  }
  int size[4] = {1,channels,height,width};
  _dst.create(4, size, CV_32F);
  float* data = (float *)_dst.data;

  for(int i=0; i<height; ++i)
    for(int j=0; j<width; ++j){
      if(channels==1){
        int offset = i*width+j;
        *(data+offset) = (static_cast<float>(src.at<uchar>(i, j))/scale - mean[0])/std[0];
      }  
      else{
        int offset = i*width+j;
        *(data+offset)                = (static_cast<float>(src.at<cv::Vec3b>(i, j)[0])/scale - mean[0])/std[0];
        *(data+offset+height*width)   = (static_cast<float>(src.at<cv::Vec3b>(i, j)[1])/scale - mean[1])/std[1];
        *(data+offset+2*height*width) = (static_cast<float>(src.at<cv::Vec3b>(i, j)[2])/scale - mean[2])/std[2];
      }
    }
}
