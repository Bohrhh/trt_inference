#include <cstdlib>
#include <cassert>
#include <sstream>
#include "stereo_camera.h"
using std::to_string;

StereoCamera::StereoCamera(const YAML::Node& cfg)
{ 
  height_ = cfg["video_paras"]["height"].as<int>();
  width_  = cfg["video_paras"]["width"].as<int>();
  rectify_alpha_  = cfg["remap_paras"]["rectify_alpha"].as<float>();

  // video 
  video_ = cfg["video"].as<std::string>();
  int id = std::atoi(video_.c_str());
  if(to_string(id) == video_)
    cap_.open(id);
  else 
    cap_.open(video_);
  assert(cap_.isOpened() && "Open video failed!");
  if(!cap_.isOpened()){
    std::cerr << "Open " << video_ << " failed!\n";
    exit(EXIT_FAILURE);
  }
  if(to_string(id) == video_){
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    const char* fourcc = cfg["video_paras"]["fourcc"].as<std::string>().c_str();
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]));
  }
  else{
    height_ = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    width_  = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
  }

  // remap or not
  remap_ = static_cast<bool>(cfg["remap"].as<int>());
  if(remap_){
    std::string calib_file = cfg["remap_paras"]["calib_file"].as<std::string>();
    load(calib_file);
  }

  // save video or not
  save_ = static_cast<bool>(cfg["save"].as<int>());
  if (save_){
    std::string name  = cfg["save_paras"]["name"].as<std::string>();
    int height  = cfg["save_paras"]["height"].as<int>();
    int width   = cfg["save_paras"]["width"].as<int>();
    double freq = cfg["save_paras"]["freq"].as<double>();
    wri_.open(
      name,
      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      freq,
      cv::Size(width, height));
  }

}

void StereoCamera::load(const std::string& filename)
{
  // load intrinsics
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  fs["cameraMatrix1"] >> cameraMatrixL_;
  fs["distCoeffs1"]   >> distCoeffsL_;
  fs["cameraMatrix2"] >> cameraMatrixR_;
  fs["distCoeffs2"]   >> distCoeffsR_;
  fs["R"] >> R_;
  fs["T"] >> T_;

  cv::Size sz(width_/2, height_);
  
  // stereo rectify
  cv::stereoRectify(
    cameraMatrixL_, distCoeffsL_, cameraMatrixR_, distCoeffsR_,
    sz, R_, T_, R1_, R2_, P1_, P2_, Q_, cv::CALIB_ZERO_DISPARITY, rectify_alpha_);

  cv::initUndistortRectifyMap(
    cameraMatrixL_, distCoeffsL_, R1_, P1_,
    sz, CV_32FC1, mapxL_, mapyL_);

  cv::initUndistortRectifyMap(
    cameraMatrixR_, distCoeffsR_, R2_, P2_,
    sz, CV_32FC1, mapxR_, mapyR_);
}

bool StereoCamera::read(cv::Mat& imgL, cv::Mat& imgR)
{
  cv::Mat img;
  int w;
  bool ret = cap_.read(img);
  if(!ret)
    return false;
  else{
    w    = img.cols;
    imgL = img(cv::Range::all(), cv::Range(0,w/2)).clone();
    imgR = img(cv::Range::all(), cv::Range(w/2,w)).clone();
    if(remap_){
      cv::remap(imgL, imgL, mapxL_, mapyL_, cv::INTER_LINEAR);
      cv::remap(imgR, imgR, mapxR_, mapyR_, cv::INTER_LINEAR);
    }
    return true;
  }
}

bool StereoCamera::read(cv::Mat& dst, bool isLeft)
{
  cv::Mat img;
  int w;
  bool ret = cap_.read(img);
  if(!ret)
    return false;
  else{
    w    = img.cols;
    if(isLeft)
      dst = img(cv::Range::all(), cv::Range(0,w/2)).clone();
    else
      dst = img(cv::Range::all(), cv::Range(w/2,w)).clone();

    if(remap_ && isLeft)
      cv::remap(dst, dst, mapxL_, mapyL_, cv::INTER_LINEAR);
    else if (remap_ && !isLeft)
      cv::remap(dst, dst, mapxR_, mapyR_, cv::INTER_LINEAR);
    return true;
  }
}

void StereoCamera::write(cv::Mat& img)
{
  wri_.write(img);
}

void StereoCamera::release()
{
  cap_.release();
  wri_.release();
}