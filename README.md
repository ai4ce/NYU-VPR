# NYU-VPR-Benchmark

This reposiroty provides the experiment code for the paper [Long-Term Visual Place Recognition Benchmark with View Direction and Data Anonymization Influences]().

## Requirements

- opencv (python version)
- opencv (C++ version)
- numpy
- sklearn
- [Pytorch](https://pytorch.org/get-started/locally/) (with specific CUDA version)
- torchvision (same with Pytorch)
- matplotlib
- pillow
- utm
- tensorboardX

## Data Processing

**1. Image Anonymization**

To install mseg-api:

```
$ cd segmentation
$ cd mseg-api
$ pip install -e .
```

Make sure that you can run `python -c "import mseg"` in python.

To install mseg-semantic:

```
$ cd segmentation
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

$ cd ../mseg-semantic
$ pip install -e .
```

Make sure that you can run `python -c "import mseg_semantic"` in python.

Finally:

```
$ input_file=/path/to/my/directory
$ model_name=mseg-3m
$ model_path=mseg_semantic/mseg-3m.pth
$ config=mseg_semantic/config/test/default_config_360_ms.yaml
$ python -u mseg_semantic/tool/universal_demo.py --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file}
```

**2. Image Filtration**

Inside the `process` folder, use `whiteFilter.py` to filter images with white pixel percentage.

## Methods

**1. VLAD+SURF**

Modify `vlad_codebook_generation.py` line 157 - 170 to fit the dataset.

```
$ cd test/vlad
$ python vlad_codebook_generation.py
$ python query_image_closest_image_generation.py
```

*Notice: the processing may take a few hours.

**2. VLAD+SuperPoint**

```
$ cd test/vlad_SP
$ python main.py
$ python find_closest.py
```

*Notice: the processing may take a few hours.

**3. NetVLAD**

**4. PoseNet**

Copy the `train_image_paths.txt` and `test_image_paths.txt` to test/posenet.

Obtain the latitude and longtitude of training images and convert them to normalized universal transverse mercator (UTM) coordinates.

```
$ cd test/posenet
$ python getGPS.py
$ python mean.py
```

Start training. This may take seceral hours. Suggestion: use slurm to run the process.

```
$ python train.py --image_path path_to_train_images/ --metadata_path trainNorm.txt
```

Generate the input file for testing from test_image_paths.txt.

```
$ python gen_test_txt.py
```

Start testing.
```
$ python single_test.py --image_path path_to_test_images/ --metadata_path test.txt --weights_path models_trainNorm/best_net.pth
```

The predicted normalized UTM coordinates of test images is in the image_name.txt. Match the test images with the training images based on their location.

```
$ python match.py
```

The matching result is in the match.txt.

**5. DBoW**

Copy the train_image_paths.txt and test_image_paths.txt to test/DBow3/utils. Copy and paste the content of test_image_paths.txt at the end of train_image_paths.txt and save the text file as total_images_paths.txt.

Open test/DBow3/utils/demo_general.cpp file. Change the for loop range at line 117 and line 123. Both ranges are the range of lines in total_images_paths.txt. The first for loop range is the range of test images and the second range is the range of training images. To run with multi-thread, you may run the code multiple times with small ranges of test images where the sum of ranges equals to the number of lines in test_image_paths.txt.

Compile and run the code.

```
$ cd test/DBow3
$ cmake .
$ cd utils
$ make
$ ./demo_general a b
```

The result of each test image and its top-5 matched training images is in the output.txt.