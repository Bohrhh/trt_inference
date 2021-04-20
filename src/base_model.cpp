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


int count(const cv::MatSize& size)
{
  int ndims = size.dims();
  int sum = 1;
  for(int i=0; i<ndims; ++i)
    sum *= size[i];
  return sum;
}


BaseModel::BaseModel(const YAML::Node& cfg)
{
  std::string modelFilename = cfg["path"].as<std::string>();
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;

  if (!cfg["inputs"].IsSequence()){
    std::cerr << "Cfg Inputs should be sequence!" << std::endl;
    exit(1);
  }
  for (std::size_t i=0; i<cfg["inputs"].size(); ++i) {
    inputTensorNames.emplace_back(cfg["inputs"][i].as<std::string>());
  }

  if (!cfg["outputs"].IsSequence()){
    std::cerr << "Cfg outputs should be sequence!" << std::endl;
    exit(1);
  }
  for (std::size_t i=0; i<cfg["outputs"].size(); ++i) {
    outputTensorNames.emplace_back(cfg["outputs"][i].as<std::string>());
  }

  bool ret = open(modelFilename, inputTensorNames, outputTensorNames);
  if (!ret) {
    std::cerr << "Model constructs failed!" << std::endl;
    exit(1);
  }
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
    std::cout << "Failed to createInferBuilder" << std::endl;
    return false;
  }

  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network)
  {
    std::cout << "Failed to createNetwork" << std::endl;
    return false;
  }

  auto parser = SampleUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  if (!parser)
  {
    std::cout << "Failed to createParser" << std::endl;
    return false;
  }

  auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
  if (!runtime)
  {
    std::cout << "Failed to createInferRuntime" << std::endl;
    return false;
  }

  std::ifstream ifs(filename_, std::ios::binary);
  if (!ifs)
  {
    std::cout << "Failed to open engine file: " << filename_ << std::endl;
    return false;
  }

  ifs.seekg(0, ifs.end);
  std::ifstream::pos_type size = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  std::vector<char> data(size);
  if (!ifs.read(data.data(), size))
  {
    std::cout << "Failed to load engine file: " << filename_ << std::endl;
    return false;
  }

  engine_ = SampleUniquePtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(data.data(), size, nullptr));
  if (!engine_)
  {
    std::cout << "Failed to deserializeCudaEngine" << std::endl;
    return false;
  }

  context_ = engine_->createExecutionContext();
  context_->setOptimizationProfile(0);

  for(std::string n : inputTensorNames_){
    int id = engine_->getBindingIndex(n.c_str());
    nvinfer1::Dims dims = engine_->getProfileDimensions(id, 0, nvinfer1::OptProfileSelector::kOPT);
    inOutDims_[n] = dims;
    context_->setBindingDimensions(id, dims);
  }
    
  for(std::string n : outputTensorNames_){
    int id = engine_->getBindingIndex(n.c_str());
    inOutDims_[n] = context_->getBindingDimensions(id);
    outputDataType_[n] = engine_->getBindingDataType(id);
  }

  buffers_ = new samplesCommon::BufferManager(engine_, 0, context_);

  return engine_.get();
}


void BaseModel::checkInputsDims(std::unordered_map<std::string, cv::Mat>& inputs)
{
  bool changed = false;
  for(std::string n : inputTensorNames_){
    cv::Mat x = inputs[n];
    cv::MatSize currentInputSize = x.size;
    nvinfer1::Dims oldInputDims = inOutDims_[n];
    int nbDims = oldInputDims.nbDims;
    for(int i=0; i<nbDims; i++){
      if(oldInputDims.d[i]!=currentInputSize[i]){
        changed = true;
        // change inOutDims_
        nvinfer1::Dims currentInputDims;
        currentInputDims.nbDims = nbDims;
        for(int i=0; i<nbDims; i++)
          currentInputDims.d[i] = currentInputSize[i];
        inOutDims_[n] = currentInputDims;
        break;
      }
    }
  }

  if(!changed)
    return ;

  for(std::string n : inputTensorNames_){
    int id = engine_->getBindingIndex(n.c_str());
    context_->setBindingDimensions(id, inOutDims_[n]);
  }
  for(std::string n : outputTensorNames_){
    int id = engine_->getBindingIndex(n.c_str());
    inOutDims_[n] = context_->getBindingDimensions(id);
  }

  delete buffers_;
  buffers_ = new samplesCommon::BufferManager(engine_, 0, context_);
}


bool BaseModel::run(
  std::unordered_map<std::string, cv::Mat>& inputs, 
  std::unordered_map<std::string, cv::Mat>& outputs)
{
  checkInputsDims(inputs);
  
  {// input
    for(std::string n : inputTensorNames_){
      float* hostDataBuffer = static_cast<float*>(buffers_->getHostBuffer(n));
      cv::Mat x = inputs[n];
      memcpy(hostDataBuffer, x.data, count(x.size)*sizeof(float));
    }
  }

  // Memcpy from host input buffers_ to device input buffers_
  buffers_->copyInputToDevice();

  bool status = context_->executeV2(buffers_->getDeviceBindings().data());
  if (!status) return false;

  // Memcpy from device output buffers_ to host output buffers_
  buffers_->copyOutputToHost();

  {// output
    for(std::string n : outputTensorNames_){
      cv::Mat y;
      nvinfer1::Dims dims = inOutDims_[n];
      if(outputDataType_[n] == nvinfer1::DataType::kFLOAT){
        y.create(dims.nbDims, dims.d, CV_32F);
        const float* src = static_cast<float*>(buffers_->getHostBuffer(n));
        int num          = count(dims); 
        memcpy(y.data, src, num*sizeof(float));
      }
      else if(outputDataType_[n] == nvinfer1::DataType::kINT32){
        y.create(dims.nbDims, dims.d, CV_32S);
        const int32_t* src = static_cast<int32_t*>(buffers_->getHostBuffer(n));
        int num            = count(dims); 
        memcpy(y.data, src, num*sizeof(int32_t));
      }
      outputs[n] = y;
   }
  }

  return true;
}