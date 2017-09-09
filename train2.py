"""
input を12チャンネルにしたversion
"""
import keras
import keras.backend as K
import numpy as np
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
from models import discriminator, generator, GAN
from facade_dataset2 import load_dataset
from PIL import Image
import math
import os


def train(patch_size, batch_size, epochs):

    def l1_loss(y_true,y_pred):
        return K.mean(K.abs(y_pred - y_true),axis=[1,2,3])

    if not os.path.exists("./result"):
        os.mkdir("./result")

    resultDir = "./result/" + "patch" + str(patch_size)
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)

    modelDir = "./model"
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)

    patch_size = patch_size
    batch_size = batch_size
    nb_epoch = epochs

    train_img, train_label = load_dataset(data_range=(1,370))
    # train_label = train_label[:,:,:,np.newaxis]
    # test_img, test_label = load_dataset(data_range=(300,379))
    # test_label = test_label[:,:,:,np.newaxis]


    # Create optimizers
    opt_gan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    gan_loss = [l1_loss, 'binary_crossentropy']
    gan_loss_weights = [100,1]

    gen = generator()

    dis = discriminator(patch_size)
    dis.trainable = False

    gan = GAN(gen,dis)
    gan.compile(loss = gan_loss, loss_weights = gan_loss_weights,optimizer = opt_gan)

    dis.trainable = True
    dis.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    nb_train = 299
    for epoch in range(nb_epoch):
        print("Epoch is ", epoch)
        print("Number of batches", int(nb_train/batch_size))
        ind = np.random.permutation(nb_train)
        for index in range(int(nb_train/batch_size)):
            print(index)
            img_batch = train_img[ind[(index*batch_size) : ((index+1)*batch_size)],:,:,:]
            label_batch =train_label[ind[(index*batch_size) : ((index+1)*batch_size)],:,:,:]
            generated_img = gen.predict(label_batch)
            if epoch % 10 == 0 and index == 0:
                image = combine_images(label_batch)
                x = np.ones((image.shape[0],image.shape[1],3)).astype(np.uint8)*255
                # x[:,:,0] = np.uint8(15*image.reshape(image.shape[0],image.shape[1]))
                x[:,:,0] = 0
                for i in range(12):
                    x[:,:,0] += np.uint8(15*i*image[:,:,i])
                Image.fromarray(x,mode="HSV").convert('RGB').save(resultDir + "/label_" + str(epoch)+"epoch.png")

                image = combine_images(img_batch)
                image = image*128.0+128.0
                Image.fromarray(image.astype(np.uint8)).save(resultDir + "/gt_" + str(epoch)+"epoch.png")

                image = combine_images(generated_img)
                image = image*128.0+128.0
                Image.fromarray(image.astype(np.uint8)).save(resultDir + "/generated_" + str(epoch)+"epoch.png")
            labels = np.concatenate([label_batch,label_batch])
            imgs = np.concatenate([img_batch,generated_img])
            dis_y = np.array([1] * batch_size + [0] * batch_size)
            d_loss = np.array(dis.train_on_batch([labels,imgs],dis_y ))
            # print("disriminator_loss : " + str(d_loss) )
            gan_y = np.array([1] * batch_size)
            g_loss = gan.train_on_batch([label_batch, img_batch], [img_batch, gan_y])
            # print("gan_loss : " + str(g_loss) )
        print("disriminator_loss : " + str(d_loss) )
        print("gan_loss : " + str(g_loss) )
    # gan.save("gan_" + "patch" + str(patch_size) + ".h5")



def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    ch = generated_images.shape[3]
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1],ch),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = \
            img[:, :, :]
    return image

if __name__ == '__main__':
    train(patch_size=64, batch_size=10, epochs=1000)
