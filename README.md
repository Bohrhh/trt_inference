# Tensorrt Inference Sample Code

## Requirements

* tensorrt==7.x
* opencv==3.4.x
* gflags
* yaml-cpp
* [tensorrt_plugin](http://115.239.209.130:1111/zhilu/dl/quantization/tensorrt_plugin)

## Directory

```
├── configs/
│   ├── camera/ camera calibration files
│   ├── models/ model config files
│
├── common/  # nvidia common utils
│
├── include/
│
├── src/
│   ├── models/
│   ├── stereo_camera.cpp
│   ├── preprocess.cpp
│   ├── base_model.cpp
│   ├── main.cpp

```

## Usage

```bash
mkdir build && cd build
cmake ..
make

cd ..
build/inference --config-file=configs/fastFeat.yaml
# settings of camera and model are in yaml file
```