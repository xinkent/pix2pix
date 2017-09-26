"""
input を12チャンネルにしたversion
"""
import keras
import keras.backend as K
import numpy as np
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
from models import discriminator, generator, GAN,discriminator2,discriminator3
from facade_dataset2 import load_dataset
from PIL import Image
import math
import os
import tensorflow as tf
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)

def train():
    parser = argparse.ArgumentParser(description = "keras pix2pix")
    parser.add_argument('--batchsize', '-b', type=int, default = 1)
    parser.add_argument('--patchsize', '-p', type=int, default = 64)
    parser.add_argument('--epoch', '-e', type=int, default = 500)
    parser.add_argument('--out', '-o',default = 'result')
    parser.add_argument('--lmd', '-l',type=int, default = 100)
    args = parser.parse_args()

    def l1_loss(y_true,y_pred):
        return K.mean(K.abs(y_pred - y_true),axis=[1,2,3])

    if not os.path.exists("./result"):
        os.mkdir("./result")

    # resultDir = "./result/" + "patch" + str(patch_size)
    resultDir = "./result/" + args.out
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)


    """
    modelDir = "./model"
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    """
    patch_size = args.patchsize
    batch_size = args.batchsize
    nb_epoch = args.epoch
    lmd = args.lmd

    o = open(resultDir + "/log","w")
    o.write("batch:" + str(batch_size) + "  lambda:" + str(lambda) + "\n")
    o.write("dis_loss,gan_mae,gan_entropy,vgan_mae,vgan_entropy" + "\n")
    o.close()

    train_img, train_label = load_dataset(data_range=(1,300))
    # train_label = train_label[:,:,:,np.newaxis]
    test_img, test_label = load_dataset(data_range=(300,379))
    # test_label = test_label[:,:,:,np.newaxis]


    # Create optimizers
    opt_gan = Adam(lr=1E-3)
    # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
    opt_discriminator = Adam(lr=1E-3)
    opt_generator = Adam(lr=1E-3)

    gan_loss = ['mae', 'binary_crossentropy']
    gan_loss_weights = [lmd,1]

    gen = generator()
    gen.compile(loss = 'mae', optimizer=opt_generator)

    dis = discriminator2()
    # dis = discriminator3(patch_size)
    dis.trainable = False

    gan = GAN(gen,dis)
    gan.compile(loss = gan_loss, loss_weights = gan_loss_weights,optimizer = opt_gan)

    dis.trainable = True
    dis.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    train_n = train_img.shape[0]
    test_n = test_img.shape[0]
    print(train_n,test_n)
    for epoch in range(nb_epoch):

        o = open(resultDir + "/log","a")
        print("Epoch is ", epoch)
        print("Number of batches", int(train_n/batch_size))
        ind = np.random.permutation(train_n)
        for index in range(int(train_n/batch_size)):
            print(index)
            img_batch = train_img[ind[(index*batch_size) : ((index+1)*batch_size)],:,:,:]
            label_batch =train_label[ind[(index*batch_size) : ((index+1)*batch_size)],:,:,:]
            generated_img = gen.predict(label_batch)

            labels = np.concatenate([label_batch,label_batch])
            imgs = np.concatenate([img_batch,generated_img])
            dis_y = np.array([1] * batch_size + [0] * batch_size)
            d_loss = np.array(dis.train_on_batch([labels,imgs],dis_y ))
            # print("disriminator_loss : " + str(d_loss) )
            gan_y = np.array([1] * batch_size)
            g_loss = gan.train_on_batch([label_batch], [img_batch, gan_y])
            # print("gan_loss : " + str(g_loss) )

            if epoch % 25 == 0 and index == 0:
                test_ind = np.random.permutation(test_n)
                test_img_batch = test_img[test_ind[0:batch_size],:,:,:]
                test_label_batch = test_label[test_ind[0:batch_size],:,:,:]
                test_generated_img = gen.predict(test_label_batch)
                validation_gan_loss = gan.test_on_batch([test_label_batch],[test_img_batch, gan_y])

                image = combine_images(test_label_batch)
                x = np.ones((image.shape[0],image.shape[1],3)).astype(np.uint8)*255
                # x[:,:,0] = np.uint8(15*image.reshape(image.shape[0],image.shape[1]))
                x[:,:,0] = 0
                for i in range(12):
                    x[:,:,0] += np.uint8(15*i*image[:,:,i])
                Image.fromarray(x,mode="HSV").convert('RGB').save(resultDir + "/vlabel_" + str(epoch)+"epoch.png")

                image = combine_images(test_img_batch)
                image = image*128.0+128.0
                Image.fromarray(image.astype(np.uint8)).save(resultDir + "/vgt_" + str(epoch)+"epoch.png")

                image = combine_images(test_generated_img)
                image = image*128.0+128.0
                Image.fromarray(image.astype(np.uint8)).save(resultDir + "/vgenerated_" + str(epoch)+"epoch.png")


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

                generated_img = gen.predict(label_batch)

                image = combine_images(generated_img)
                image = image*128.0+128.0
                Image.fromarray(image.astype(np.uint8)).save(resultDir + "/generated_" + str(epoch)+"epoch.png")
                o.write(str(epoch) + "," + str(d_loss), "," + str(g_loss[0]) + "," + str(g_loss[1]) + "," + str(validate_gan_loss[0]) +"," + str(validate_gan_loss[1]) + "\n")
        o.close()
    # o.close()
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
    train()
