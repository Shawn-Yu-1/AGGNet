import os 
from tqdm import tqdm
from PIL import Image

path = "/data/dataset/places512/data_large/"
fnames = []

for root, _, files in os.walk(path):
    for f in files:
        fname = os.path.join(root, f)
        fnames.append(fname)

## resize the dataset to a fixed size
## you can do it by train.       
for fname in tqdm(fnames):
    img = Image.open(fname).convert("RGB")
    w, h = img.size
    if min(w, h) < 512:
        rate = 520 / min(w, h)
        img = img.resize((int(w*rate), int(h*rate))) 
    w, h = img.size
    left = (w - 512) // 2
    up = (h - 512) // 2
    out = img.crop((left, up, left+512, up+512)) 
    out.save(fname.replace("/data_large/", "/train_512/"))      