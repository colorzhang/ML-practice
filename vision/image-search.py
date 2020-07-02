# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h2>Model and predictions</>

# %%
import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image


# %%
#Import Resnet50 model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3),pooling='avg')

model.summary()


# %%
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img, data_format='channels_last')
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, data_format='channels_last')
    preds = model.predict(x)
    return img_path, preds


# %%
preds = predict('data/feidegger/fashion/0VB21C000-A11@12.1.jpg')

preds[1].shape


# %%


# %% [markdown]
# <h2>Build knn index</>

# %%
# setting up the Elasticsearch connection
from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['localhost:9200'],
    http_auth=('admin', 'admin'),
    use_ssl=True,
    verify_certs=False
)


# %%
#Define KNN Elasticsearch index maping
knn_index = {
    "settings": {
        "index.knn": True
    },
    "mappings": {
        "properties": {
            "zalando_img_vector": {
                "type": "knn_vector",
                "dimension": 2048
            }
        }
    }
}

#Creating the Elasticsearch index
es.indices.create(index="idx_zalando", body=knn_index, ignore=400)


# %%
es.indices.get(index="idx_zalando")


# %%
# defining a function to import the feature vectors corrosponds to each S3 URI into Elasticsearch KNN index
# This process will take around ~3 min.


def es_import(i):
    return es.index(index='idx_zalando', 
             body={"zalando_img_vector": i[1][0], "image": i[0]}
            )
    
# preds[1]
es_import(preds)


# %%
res = es.search(index="idx_zalando", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total']['value'])
# for hit in res['hits']['hits']:
#     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])
# print(res)

# %% [markdown]
# <h2>Build all image index</>

# %%
base_dir = 'data/feidegger/fashion'
all_images = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
print(len(all_images))

# all_images = all_images[:10]
# all_images[0]


# %%
# all_images = all_images[:20]
all_features = list(map(predict, all_images))
print(len(all_features))
# all_features[0][0]
# all_features[0][1]


# %%
r= map(es_import, all_features)
print (len(list(r)))
# es_import(all_features[0])


# %%
res = es.delete_by_query(index="idx_zalando", body={"query": {"match_all": {}}})


# %%
sample_imgpath = 'data/feidegger/fashion/DE121C0DV-Q11@14.jpg'
display(Image.open(sample_imgpath))
sample_features = predict(sample_imgpath)[1][0]


# %%
import json
k = 10
idx_name = 'idx_zalando'
res = es.search(request_timeout=30, index=idx_name,
                body={'size': k, 
                      'query': {'knn': {'zalando_img_vector': {'vector': sample_features, 'k': k}}}})


# %%
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 8), subplot_kw={'xticks': [], 'yticks': []})

for i in range(5):
    for j in range(2):
        key = res['hits']['hits'][(i+1)*(j+1)-1]['_source']['image']
        img = mpimg.imread(key)
        axs[j,i].imshow(img)
plt.tight_layout()
plt.show()


# %%


