import sys
import os
import tensorflow as tf
import random
import numpy as np
import sys
import os
from vaegan.VAEGAN import *
import pdb

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from utils import PARSER
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
def ds_gen():
    dirname = 'results/{}/record'.format(args.env_name)
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                img_i = img / 255.0
                yield img_i

def save_model(vaegan):
    tf.keras.models.save_model(vaegan.E, model_save_path+'_E_', 
        include_optimizer=True, save_format='tf')
    tf.keras.models.save_model(vaegan.G, model_save_path+'_G_', 
        include_optimizer=True, save_format='tf')
    tf.keras.models.save_model(vaegan.D, model_save_path+'_D_', 
        include_optimizer=True, save_format='tf')


def log_images(vaegan,step,args,test_images) :
    lattent,_ = vaegan.E(test_images)
    fake = vaegan.G(lattent)
    fake_r = vaegan.G(test_r)
    tf.summary.image("reconstructed image", fake[:8], step=step, max_outputs=4)
    tf.summary.image("random image", fake_r[:8], step=step, max_outputs=4)
    dis_fake,inner_dis_fake = vaegan.D(fake)
    dis_fake_r,inner_dis_fake_r = vaegan.D(fake_r)
    dis_true,inner_dis_true = vaegan.D(test_images)
    tf.summary.histogram("dis fake", inner_dis_fake, step=step, buckets=20)
    tf.summary.histogram("dis true", inner_dis_true, step=step, buckets=20)
    tf.summary.histogram("dis random", inner_dis_fake_r, step=step, buckets=20)
    tf.summary.histogram("dis lattent", lattent, step=step, buckets=20)
    tf.summary.histogram("dis normal", tf.random.normal((args.vae_gan_batch_size, args.vae_gan_LATENT_DEPTH)), step=step, buckets=20) 


if __name__ == "__main__": 
    model_save_path = "results/{}/tf_vaegan".format(args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)
    shuffle_size = 20 * 1000 # only loads ~20 episodes for shuffle windows b/c im poor and don't have much RAM
    ds = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(args.vae_batch_size)
    ds = ds.prefetch(100) # prefetch 100 batches in the buffer #tf.data.experimental.AUTOTUNE)
    


    vaegan = VAEGAN(args=args)
    #tensorboard_callback.set_model(vaegan)

    step = 0
    log_freq,img_log_freq = 100, 100
    save_freq = 1000
    vip_loss_names =['E_loss','G_loss','D_losss'] 

    for i in range(args.vae_gan_num_epoch):
        j = 0
        for x_batch in ds:
            if i == 0 and j == 0:
                vaegan._set_inputs(x_batch)
            j += 1
            step += 1 
            
            vip_loss,detailed_loss = vaegan.train_step_vaegan(x_batch)
            [tf.summary.scalar(loss_key,loss_val.numpy(), step=step) for loss_key, loss_val in zip(vip_loss_names,vip_loss)] 

            if j % log_freq == 0:
                output_log = 'epoch: {} mb: {}'.format(i, j)
                for loss_key,loss_val in zip(vip_loss_names,vip_loss):
                    output_log += ', {}: {:.4f}'.format(loss_key, loss_val.numpy())
                print(output_log)

            if j % img_log_freq == 0:
                log_images(vaegan,step,args,test_images=x_batch[0])

            if j % save_freq == 0:
                save_model(vaegan)
                print('saving')
    
    save_model(vaegan)
    print('final saving')
    
