import json
import re
from transformers import BertModel
from transformers import BertTokenizer
import torch
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from collections import OrderedDict, Counter
import math
from numpy import asarray
from numpy import save
from numpy import load


def resize_and_crop_image(input_file, output_box=[224, 224], fit=True):
        # https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
        '''Downsample the image.
        '''
        img = Image.open(input_file)
        #img.save("orig_"+input_file.split('/')[-1])
        box = output_box
        # preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
            factor *= 2
        if factor > 1:
            img.thumbnail(
                (img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

        # calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2 / box[0]
            hRatio = 1.0 * y2 / box[1]
            if hRatio > wRatio:
                y1 = int(y2 / 2 - box[1] * wRatio / 2)
                y2 = int(y2 / 2 + box[1] * wRatio / 2)
            else:
                x1 = int(x2 / 2 - box[0] * hRatio / 2)
                x2 = int(x2 / 2 + box[0] * hRatio / 2)
            img = img.crop((x1, y1, x2, y2))

        # Resize the image with best quality algorithm ANTI-ALIAS
        img = img.resize(box, Image.ANTIALIAS).convert('RGB')
        #img = numpy.asarray(img, dtype='float32')
        return img
    

def get_image_feature(feature_extractor, image):
    with torch.no_grad():
        feature_images = feature_extractor.features(image)
        feature_images = feature_extractor.avgpool(feature_images)
        feature_images = torch.flatten(feature_images, 1)
        feature_images = feature_extractor.classifier[0](feature_images)
    
    return feature_images


def extract_visual(img_name):
    img = resize_and_crop_image(f"/001/usuarios/isaac.bribiesca/mmimdb/dataset/{img_name}.jpeg", (256,256))
    img = preprocess(img)
    img = img.unsqueeze(0)
    feature = get_image_feature(feature_extractor, img)
    
    return feature


def extract_embedding(txt):
    text_encoded = tokenizer.encode_plus(
                txt,
                add_special_tokens=True,
                max_length=200,
                return_token_type_ids=False,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
    
    bert_vects, _ = bert(
                input_ids=text_encoded['input_ids'],
                attention_mask=text_encoded['attention_mask']
            )
    
    return bert_vects.squeeze(0).detach()


def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
with open('list.txt', 'r') as f:
    files = f.read().splitlines()
    
bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)


genres = []
ids = []
txt_f = []

for i, file in enumerate(files):
    with open(file) as f:
        data = json.load(f)
        index = file.split('/')[-1].split('.')[0]
        ids.append(index)
        genres.append(data['genres'])
        txt_f.append(data['plot'])

    if i%500==0:
        print("reading {0} ...".format(i))


n_classes = 23
counts = OrderedDict(
    Counter([g for m in genres for g in m]).most_common())
target_names = list(counts.keys())[:n_classes]


le = MultiLabelBinarizer()
Y = le.fit_transform([m for m in genres])
labels = np.nonzero(le.transform([[t] for t in target_names]))[1]


B = np.copy(Y)
rng = np.random.RandomState([2014, 8, 6])
train_idx, dev_idx, test_idx = [], [], []
test_size = 0.3
dev_size = 0.1
for l in labels[::-1]:
    t = B[:, l].nonzero()[0]
    t = rng.permutation(t)
    n_test = int(math.ceil(len(t) * test_size))
    n_dev = int(math.ceil(len(t) * dev_size))
    n_train = len(t) - n_test - n_dev
    test_idx.extend(t[:n_test])
    dev_idx.extend(t[n_test:n_test + n_dev])
    train_idx.extend(t[n_test + n_dev:])
    B[t, :] = 0


indices = np.concatenate([train_idx, dev_idx, test_idx])
nsamples = len(indices)
nsamples_train, nsamples_dev, nsamples_test = len(
    train_idx), len(dev_idx), len(test_idx)


train_idx = [int(i) for i in train_idx]
dev_idx = [int(i) for i in dev_idx]
test_idx = [int(i) for i in test_idx]

max_len=120
dict_idxs = {'train': list(np.array(ids)[train_idx]), 'dev': list(np.array(ids)[dev_idx]), 'test': list(np.array(ids)[test_idx])}
with open('../mmimdb/data_bert_'+str(max_len)+'/partition.json', 'w') as fp:
    json.dump(dict_idxs, fp)


for i, idx in enumerate(ids):
    txt_tensor = extract_embedding(txt_f[i])
    img_tensor = extract_visual(idx)
    
    example = {'txt': txt_tensor, 'img': img_tensor, 'labels': torch.tensor(Y[i,labels])}
    torch.save(example, f'../mmimdb/data_bert_'+str(max_len)+f'/features/{idx}.pt')
    
    if i%300==0:
        print(f"Saving {i} ...")