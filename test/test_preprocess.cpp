#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <unordered_map>
#include <stdexcept>
#include <sys/stat.h>

#include "stereo_camera.h" 
#include "preprocess.h"

bool exists(const std::string& path){
  struct stat statbuf;
  return stat(path.c_str(), &statbuf) == 0;
}

int main(int argc, char **argv)
{
  if(argc!=2){
    std::cout << "Please run command: ./inference path/to/config.yaml" << std::endl;
    return -1;
  }else if(std::string(argv[1])=="-h" || std::string(argv[1])=="--help"){
    std::cout << "Please run command: ./inference path/to/config.yaml" << std::endl;
    return -1;
  }else if(!exists(argv[1])){
    std::cout << "No such file: " << argv[1] << std::endl;
    std::cout << "Please run command: ./inference path/to/config.yaml" << std::endl;
    return -1;
  }

  YAML::Node cfg = YAML::LoadFile(argv[1]);
  
  StereoCamera camera(cfg["stereo_camera"]);
  Preprocess pre(cfg["preprocess"]);
  
  // video or camera inference
  while (true)
  {
    cv::Mat imgL;
    cv::Mat imgR;
    cv::Mat x1;
    cv::Mat x2;

    bool ret_cap = camera.read(imgL, imgR);
    if(!ret_cap)
      break;

    pre.resize(imgL, x1, 320, 512, imgPre::ResizeMode::Bilinear, imgPre::PaddingMode::LetterBox, 128);

    cv::imshow("vis", x1);

    int key = cv::waitKey(1);
    if(key == 'q' || key == 27 /* ESC */)
      break;
  }

  cv::destroyAllWindows();

  return 0;
}
