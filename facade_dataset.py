import numpy as np
from PIL import Image

from io import BytesIO


def load_dataset(dataDir='./dataset/base/', data_range=(1,300)):
        print("load dataset start")
        print("     from: %s"%dataDir)
        img_dataset = []
        label_dataset = []
        for i in range(data_range[0],data_range[1]):
            img = Image.open(dataDir + "/cmp_b0001.jpg")
            label = Image.open(dataDir + "/cmp_b0001.png")
            w,h = img.size
            r = 286/min(w,h)
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label = label.resize((int(r*w), int(r*h)),Image.NEAREST)

            img = np.asarray(img).astype("f")
            img = np.asarray(img).astype("f")/128.0-1.0

            label = np.asarray(label)-1

            img_h,img_w,_ = img.shape
            label_h, label_w = label.shape
            img_xl = np.random.randint(0,img_w-256)
            img_yl = np.random.randint(0,img_h-256)
            label_xl = np.random.randint(0,label_w-256)
            label_yl = np.random.randint(0,label_h-256)
            img = img[img_yl:img_yl+256, img_xl:img_xl+256,:]
            label = label[label_yl:label_yl+256, label_xl:label_xl+256]
            img_dataset.append(img)
            label_dataset.append(label)

        print("load dataset done")
        return np.array(img_dataset),np.array(label_dataset)
