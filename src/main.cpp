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
#include "stereo_depth.h"
#include "instance_seg.h"
#include "feature_extraction.h"
#include "mono_depth.h"
#include "yolov5.h"

bool exists(const std::string& path){
  struct stat statbuf;
  return stat(path.c_str(), &statbuf) == 0;
}

int main(int argc, char **argv)
{

  // ================================================================
  // parsing args
  if(argc!=2){
    std::cerr << "Please run command: ./build/inference path/to/config.yaml" << std::endl;
    return -1;
  }else if(std::string(argv[1])=="-h" || std::string(argv[1])=="--help"){
    std::cerr << "Please run command: ./build/inference path/to/config.yaml" << std::endl;
    return -1;
  }else if(!exists(argv[1])){
    std::cerr << "No such file: " << argv[1] << std::endl;
    std::cerr << "Please run command: ./build/inference path/to/config.yaml" << std::endl;
    return -1;
  }

  // ================================================================
  // construct three main objects: camera, pre and model
  YAML::Node cfg = YAML::LoadFile(argv[1]);
  StereoCamera camera(cfg["stereoCamera"]);
  Preprocess pre(cfg["preprocess"]);
  std::shared_ptr<BaseModel> pmodel;

  bool saveVideo = static_cast<bool>(cfg["stereoCamera"]["save"].as<int>());
  std::string model_type = cfg["model"]["type"].as<std::string>();
  if(model_type=="stereo")
    pmodel = std::shared_ptr<BaseModel>(new StereoDepth(cfg["model"]));
  else if(model_type=="instanceSeg")
    pmodel = std::shared_ptr<BaseModel>(new InstanceSeg(cfg["model"]));
  else if(model_type=="featureExtraction")
    pmodel = std::shared_ptr<BaseModel>(new FeatureExtraction(cfg["model"]));
  else if(model_type=="mono")
    pmodel = std::shared_ptr<BaseModel>(new MonoDepth(cfg["model"]));
  else if(model_type=="yolov5")
    pmodel = std::shared_ptr<BaseModel>(new YOLOV5(cfg["model"]));
  else
    throw std::runtime_error("No such model type!");

  // ================================================================
  // do inference
  while (true)
  {
    cv::Mat imgL;
    cv::Mat imgR;
    cv::Mat x1;
    cv::Mat x2;
    cv::Mat visImg;
    std::unordered_map<std::string, cv::Mat> inputs;
    std::unordered_map<std::string, cv::Mat> outputs;

    bool ret_cap = camera.read(imgL, imgR);
    if(!ret_cap)
      break;

    if(model_type=="stereo"){
      pre.makePreprocess(imgL, x1);
      inputs["imgl"] = x1;
      pre.makePreprocess(imgR, x2);
      inputs["imgr"] = x2;
    }
    else{
      pre.makePreprocess(imgL, x1);
      inputs["img"] = x1;
    }

    auto start = std::chrono::system_clock::now();
    bool ret_model = pmodel->run(inputs, outputs);
    if (!ret_model){
      std::cerr << "Model run failed!" << std::endl;
      return -1;
    }
    pmodel->vis(imgL, outputs, cfg["preprocess"], visImg);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "one prediction time cost : " << diff.count()*1000 << " ms\n";
    cv::imshow("vis", visImg);
    if (saveVideo)
      camera.write(visImg);

    int key = cv::waitKey(1);
    if(key == 'q' || key == 27 /* ESC */)
      break;
  }

  camera.release();
  cv::destroyAllWindows();


  return 0;
}
