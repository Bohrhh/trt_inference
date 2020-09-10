#include "base_model.h"
#include <cassert>


int count(const nvinfer1::Dims& dim)
{
  int ndims = dim.nbDims;
  int sum = 1;
  for(int i=0; i<ndims; ++i)
    sum *= dim.d[i];
  return sum;
}

BaseModel::BaseModel(const YAML::Node& cfg)
{
  std::string modelFilename = cfg["path"].as<std::string>();
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;

  assert(cfg["inputs"].IsSequence());
  for (std::size_t i=0; i<cfg["inputs"].size(); ++i) {
    inputTensorNames.emplace_back(cfg["inputs"][i].as<std::string>());
  }

  assert(cfg["outputs"].IsSequence());
  for (std::size_t i=0; i<cfg["outputs"].size(); ++i) {
    outputTensorNames.emplace_back(cfg["outputs"][i].as<std::string>());
  }

  bool ret = open(modelFilename, inputTensorNames, outputTensorNames);
  assert(ret && "Model constructs failed!");
}


bool BaseModel::open(
    const std::string& modelFilename,
    const std::vector<std::string>& inputTensorNames,
    const std::vector<std::string>& outputTensorNames)
{
  filename_          = modelFilename;
  inputTensorNames_  = inputTensorNames;
  outputTensorNames_ = outputTensorNames;
  engine_            = nullptr;

  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder)
  {
    std::cout << "Fail to createInferBuilder" << std::endl;
    return false;
  }

  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetwork());
  if (!network)
  {
    std::cout << "Fail to createNetwork" << std::endl;
    return false;
  }

  auto parser = SampleUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  if (!parser)
  {
    std::cout << "Fail to createParser" << std::endl;
    return false;
  }

  auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
  if (!runtime)
  {
    std::cout << "Fail to createInferRuntime" << std::endl;
    return false;
  }

  std::ifstream ifs(filename_, std::ios::binary);
  if (!ifs)
  {
    std::cout << "Error opening engine file: " << filename_ << std::endl;
    return false;
  }

  ifs.seekg(0, ifs.end);
  std::ifstream::pos_type size = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  std::vector<char> data(size);
  if (!ifs.read(data.data(), size))
  {
    std::cout << "Error loading engine file: " << filename_ << std::endl;
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(data.data(), size, nullptr),
      samplesCommon::InferDeleter());
  if (!engine_)
  {
    std::cout << "Fail to deserializeCudaEngine" << std::endl;
    return false;
  }

  buffers_ = std::unique_ptr<samplesCommon::BufferManager>(new samplesCommon::BufferManager(engine_));
  context_ = SampleUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

  for(std::string n : inputTensorNames_){
    int id = engine_->getBindingIndex(n.c_str());
    inOutDims_[n] = engine_->getBindingDimensions(id);
  }
    
  for(std::string n : outputTensorNames_){
    int id = engine_->getBindingIndex(n.c_str());
    inOutDims_[n] = engine_->getBindingDimensions(id);
    outputDataType_[n] = engine_->getBindingDataType(id);
  }

  return engine_.get();
}

bool BaseModel::run(
  std::unordered_map<std::string, cv::Mat>& inputs, 
  std::unordered_map<std::string, cv::Mat>& outputs)
{
  if (!context_) return false;
  
  {// input
    for(std::string n : inputTensorNames_){
      float* hostDataBuffer_ = static_cast<float*>(buffers_->getHostBuffer(n));
      cv::Mat x = inputs[n];
      const float* data = (const float*)(x.data);
      int height        = x.rows;
      int width         = x.cols;
      int channels      = x.channels();
      assert(channels==1 || channels==3);
      if(channels==1){
        for(int i=0; i<height; ++i)
          for(int j=0; j<width; ++j)
            hostDataBuffer_[i*width+j] = x.at<float>(i,j);
      }
      else if(channels==3){
        for(int i=0; i<height; ++i)
          for(int j=0; j<width; ++j){
            hostDataBuffer_[i*width+j]                = x.at<cv::Vec3f>(i,j)[0];
            hostDataBuffer_[width*height+i*width+j]   = x.at<cv::Vec3f>(i,j)[1];
            hostDataBuffer_[2*width*height+i*width+j] = x.at<cv::Vec3f>(i,j)[2];
          }
      }
    }
  }

  // Memcpy from host input buffers_ device input buffers_
  buffers_->copyInputToDevice();

  bool status = context_->executeV2(buffers_->getDeviceBindings().data());
  if (!status) return false;

  // Memcpy from device output buffers_ host output buffers_
  buffers_->copyOutputToHost();

  {// output
    for(std::string n : outputTensorNames_){
      cv::Mat y;
      nvinfer1::Dims dims = inOutDims_[n];
      if(outputDataType_[n] == nvinfer1::DataType::kFLOAT){
        y.create(dims.nbDims, dims.d, CV_32F);
        const float* src = static_cast<float*>(buffers_->getHostBuffer(n));
        float* data      = (float*)(y.data);
        int num          = count(dims); 
        for(int j=0; j<num; ++j)
          data[j] = src[j];
      }
      else if(outputDataType_[n] == nvinfer1::DataType::kINT32){
        y.create(dims.nbDims, dims.d, CV_32S);
        const int32_t* src = static_cast<int32_t*>(buffers_->getHostBuffer(n));
        int32_t* data      = (int32_t*)(y.data);
        int num            = count(dims); 
        for(int j=0; j<num; ++j)
          data[j] = src[j];
      }
      outputs[n] = y;
   }
  }

  return true;
}