#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <gflags/gflags.h>
#include <unordered_map>

#include "stereo_camera.h" 
#include "preprocess.h"
#include "stereo_depth.h"
#include "instance_seg.h"
#include "feature_extraction.h"
#include "mono_depth.h"


DEFINE_string(config_file, "./configs/fastFeat.yaml", "path of config file");

int main(int argc, char **argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  YAML::Node cfg = YAML::LoadFile(FLAGS_config_file.c_str());
  
  StereoCamera camera(cfg["stereo_camera"]);
  Preprocess pre(cfg["preprocess"]);
  std::shared_ptr<BaseModel> pmodel;

  // construct model
  std::string model_type = cfg["model"]["type"].as<std::string>();
  if(model_type=="stereo")
    pmodel = std::shared_ptr<BaseModel>(new StereoDepth(cfg["model"]));
  else if(model_type=="instance_seg")
    pmodel = std::shared_ptr<BaseModel>(new InstanceSeg(cfg["model"]));
  else if(model_type=="feature_extraction")
    pmodel = std::shared_ptr<BaseModel>(new FeatureExtraction(cfg["model"]));
  else if(model_type=="mono")
    pmodel = std::shared_ptr<BaseModel>(new MonoDepth(cfg["model"]));
  else
    assert(false && "No such model type!");
  
  
  while (true)
  {
    cv::Mat imgL;
    cv::Mat imgR;
    cv::Mat x1;
    cv::Mat x2;
    std::unordered_map<std::string, cv::Mat> inputs;
    std::unordered_map<std::string, cv::Mat> outputs;

    bool ret_cap = camera.read(imgL, imgR);
    if(!ret_cap)
      break;

    if(model_type=="stereo"){
      pre.make_preprocess(imgL, x1);
      inputs["imgl"] = x1;
      pre.make_preprocess(imgR, x2);
      inputs["imgr"] = x2;
    }
    else{
      pre.make_preprocess(imgL, x1);
      inputs["img"] = x1;
    }

    auto start = std::chrono::system_clock::now();
    bool ret_model = pmodel->run(inputs, outputs);
    assert(ret_model && "Model run failed!");
    pmodel->vis(imgL, outputs);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "one prediction time cost : " << diff.count()*1000 << " ms\n";

    cv::imshow("vis", imgL);

    int key = cv::waitKey(1);
    if(key == 'q' || key == 27 /* ESC */)
      break;
  }

  cv::destroyAllWindows();

  return 0;
}
