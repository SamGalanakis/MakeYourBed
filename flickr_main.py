from flickr_urls import get_urls
from flickr_download import download_images
import os
import time

all_species = ['unmade bed']
images_per_species = 1000

def download():
    for species in all_species:

        print('Getting urls for', species)
        urls = get_urls(species, images_per_species)
        
        print('Downloading images for', species)
        path = os.path.join('data', species)

        download_images(urls, path)

if __name__=='__main__':

    start_time = time.time()

    download()

    print('Took', round(time.time() - start_time, 2), 'seconds')