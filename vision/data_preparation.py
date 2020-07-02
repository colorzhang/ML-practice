## Data Preparation

import os 
import shutil
import json
import tqdm
import urllib.request
from multiprocessing import Pool, freeze_support

images_path = 'data/feidegger/fashion'
filename = 'metadata.json'

if not os.path.isdir(images_path):
    os.makedirs(images_path)

def download_metadata(url):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        
#download metadata.json to local notebook
download_metadata('https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json')

def generate_image_list(filename):
    metadata = open(filename,'r')
    data = json.load(metadata)
    url_lst = []
    for i in range(len(data)):
        url_lst.append(data[i]['url'])
    return url_lst


def download_image(url):
    urllib.request.urlretrieve(url, images_path + '/' + url.split("/")[-1])
                    
def download_all():
    #generate image list            
    url_lst = generate_image_list(filename) 
    print('total urls: {}'.format(len(url_lst)))    

    workers = 2 * 4

    #downloading images to local disk
    with Pool(workers) as p:
        p.map(download_image, url_lst)


if __name__ == '__main__':
    freeze_support()
    download_all()
