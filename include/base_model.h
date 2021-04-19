#ifndef __BASE_MODEL_H__
#define __BASE_MODEL_H__

#include "buffers.h"
#include "NvOnnxParser.h"
#include "preprocess.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

class BaseModel
{
public:
  BaseModel(){};
  BaseModel(const YAML::Node& cfg);
  bool open(
    const std::string& modelFilename,
    const std::vector<std::string>& inputTensorNames = {},
    const std::vector<std::string>& outputTensorNames = {});

  bool isOpened() const { return engine_.get(); }
  void release() { engine_ = nullptr; }
  
  void checkInputsDims(
    std::unordered_map<std::string, cv::Mat>& inputs);
  bool run(
    std::unordered_map<std::string, cv::Mat>& inputs, 
    std::unordered_map<std::string, cv::Mat>& outputs);

  virtual void vis(cv::Mat& img, std::unordered_map<std::string, cv::Mat>& outputs, const YAML::Node& cfg_preprocess) = 0;

  std::unordered_map<std::string, nvinfer1::Dims> inOutDims_;
  std::unordered_map<std::string, nvinfer1::DataType> outputDataType_;
  
private:
  std::string filename_;
  std::vector<std::string> inputTensorNames_;
  std::vector<std::string> outputTensorNames_;

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  nvinfer1::IExecutionContext* context_;
  samplesCommon::BufferManager* buffers_;
};



#endif // __BASE_MODEL_H__