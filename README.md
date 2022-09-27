# CTAISegmentationCNN
Component Tree Attribute as Image for Segmentation using Convolutional Neural Network

## Reproducible Research

* Platform : Ubuntu 18.04.6 LTS
* Language : Python 3.6.9
  * TensorFlow 2.6.2

### Get code and data
Get this repository :
```
git clone git@github.com:Cyril-Meyer/CTAISegmentationCNN.git
cd CTAISegmentationCNN
```
Get the data :
```
git clone git@github.com:Cyril-Meyer/DGMM2022-MEYER-DATA.git
```
Get the source code of ComponentTreeAttributeImage :
```
git clone git@github.com:Cyril-Meyer/ComponentTreeAttributeImage.git
```
Optional : if you preferred to download the data in another folder,
create a symbolic link :
```
ln -s /Data/DGMM2022-MEYER-DATA DGMM2022-MEYER-DATA
```

### Prepare Python venv
```
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install numpy scipy scikit-image
```

If you are using the same version of Python, you should have the same
packages installed as in requirements.txt
If you have problems, check that you have the same libraries installed
as in the requirements.txt file  
*Note: if you cannot get TensorFlow 2, check that you correctly updated pip.*

### Pre process the data
*Note: all the preprocessing scripts use fixed path and no arguments.*

1. Low pass filter (9x9 mean filter) on the image slices

```
python 01_mean_filter.py
```

2. Component Tree Attribute as Image

Build ComponentTreeAttributeImage :
```
cd ComponentTreeAttributeImage/scripts
g++ -O3 -I../. CTAISegmentationCNN.cpp -o CTAISegmentationCNN
cd ../..
ln -s ComponentTreeAttributeImage/scripts/CTAISegmentationCNN 02_preprocess_ComponentTreeAttributeImage
```

Extract attribute image from pre processed images (use the script, not the binary build) :
```
./02_attribute_image.sh
```

3. Crop the image to the annotated area and construct tiff stack.

```
python 03_crop_roi.py
```

### Segmentation using Convolutional Neural Network
