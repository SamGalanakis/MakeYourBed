# MakeYourBed

A binary image classifier which tells you whether your bed is made or not. Probably not the next big app but was interesting to make!

## Data collection
Since I couldn't find any dataset for this project (unsuprisingly) I used a combination of google images and flickr api to build a dataset. 

 ### Google images

Downloaded through the python package [google-images-download](https://github.com/hardikvasa/google-images-download) in getGoogleImages.py. Requires [chromedriver](https://chromedriver.chromium.org/) to scrape any large number of images. 

The following keyword were used : 

* Made bed: "bedroom bed", "made bed", "perfect bed", "simple hotel bed"
* Unmade bed: "bedroom bed messy", "crumpled bed messy", "crumpled duved bed", "messy bed"

Eeach yielding ~  400 images.

### Flickr

Requires a Flickr api key but it's very easy to get, done in flickr_main.py. Flickr api allows for easy download of 1000+ images given a keyword and is a good source of images for such projects. This [article](https://medium.com/@adrianmrit/creating-simple-image-datasets-with-flickr-api-2f19c164d82f) explains the process.

The following keyword were used : 

* "made bed"
* "messy bed"
* "unmade bed"


The images were then manually checked, deleting irrelevant or unsuitable images. A lot of images were deleted where the bed was not the focus of the image. The flickr images had a lot of irrelevant images but that is also probably due ot the large amount of images taken for a given keyword (1000+ vs 400 for google). After going through the images I was left with ~1500 images of each class which should be enough for a half descent result. The full dataset is too large for github but will be provided if someone is interested.


## Model

Due to the limited computational resources at my disposal I chose to use transfer learning with a model pretrained on imagenet rather than training it from scratch. Specifically, resnet-101 as provided by the convenient pytorch based package [cnn-finetune](https://pypi.org/project/cnn-finetune/). Only the classifier (fully connected layers) was changed in order to be appropriate for binary classification. The images were resized to be 224x224 before being inputed but a smaller image size may have been sufficient.

## Training



Without using any image augmentation the model quicly fits to the train data, approaching 100% accuracy within 22 epochs but the test accuracy does not increase past 12 epocks (88% test accuracy) so training past that point seems to start overfitting to the train data.
![](graphics\base_training.png)





In order to avoid overfitting and possibly increase generalization I tried to use image augmentation, spepcifically random rotations, flips and [ColorJitter](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ColorJitter). I ran training for significantly longer time and the model did take a while to deal with all the variability in the train set but eventually reached a similar test performance to the previous model so the augmentation does not seem to have helped much. It is likely that the model can not do better than 88% due to the low quality of the dataset and test set despite manual filtering.

![](graphics\augmentet_training.png)