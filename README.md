# MakeYourBed

A binary image classifier which tells you whether your bed is made or not. Probably not the next big app but was interesting to make!

## Data collection:
Since I couldn't find any dataset for this project (unsuprisingly) I used a combination of google images and flickr api to build a dataset. 

 ### Google images: 

Downloaded through the python package [google-images-download](https://github.com/hardikvasa/google-images-download) in getGoogleImages.py. Requires [chromedriver](https://chromedriver.chromium.org/) to scrape any large number of images. 

The following keyword were used : 

* Made bed: "bedroom bed", "made bed", "perfect bed", "simple hotel bed"
* Unmade bed: "bedroom bed messy", "crumpled bed messy", "crumpled duved bed", "messy bed"

Eeach yielding ~  400 images.

### Flickr:

Requires a Flickr api key but it's very easy to get, done in flickr_main.py. Flickr api allows for easy download of 1000+ images given a keyword and is a good source of images for such projects. This [article](https://medium.com/@adrianmrit/creating-simple-image-datasets-with-flickr-api-2f19c164d82f) explains the process.

The following keyword were used : 

* "made bed"
* "messy bed"
* "unmade bed"


flickr : https://medium.com/@adrianmrit/creating-simple-image-datasets-with-flickr-api-2f19c164d82f

google images download : https://github.com/hardikvasa/google-images-download