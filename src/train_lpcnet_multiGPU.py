'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Train a LPCNet model (note not a Wavenet model)

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import Callback, ReduceLROnPlateau

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()

# use this option to reserve GPU memory, e.g. for running more than
# one thing at a time.  Best to disable for GPUs with small memory
config.gpu_options.per_process_gpu_memory_fraction = 0.83
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = False
config.allow_soft_placement = True
set_session(tf.Session(config=config))

init_epoch = 0
nb_epochs = 100

# Try reducing batch_size if you run out of memory on your GPU
batch_size = 256

#with tf.device("/gpu:0"):
model, _, _ = lpcnet.new_lpcnet_model(training=True, use_gpu=True)

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# with tf.device("/gpu:0"):
if True:
    feature_file = sys.argv[1]
    pcm_file = sys.argv[2]     # 16 bit unsigned short PCM samples
    frame_size = model.frame_size
    nb_features = 55
    nb_used_features = model.nb_used_features
    feature_chunk_size = 15
    pcm_chunk_size = frame_size*feature_chunk_size

    # u for unquantised, load 16 bit PCM samples and convert to mu-law

    data = np.fromfile(pcm_file, dtype='uint8')
    nb_frames = len(data)//(4*pcm_chunk_size)

    features = np.fromfile(feature_file, dtype='float32')

    # limit to discrete number of frames
    data = data[:nb_frames*4*pcm_chunk_size]
    features = features[:nb_frames*feature_chunk_size*nb_features]

    features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))

    sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
    pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
    in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
    out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))
    del data

    print("ulaw std = ", np.std(out_exc))

    features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
    features = features[:, :, :nb_used_features]
    features[:,:,18:36] = 0

    fpad1 = np.concatenate([features[0:1, 0:2, :], features[:-1, -2:, :]], axis=0)
    fpad2 = np.concatenate([features[1:, :2, :], features[0:1, -2:, :]], axis=0)
    features = np.concatenate([fpad1, features, fpad2], axis=1)


    periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')

    in_data = np.concatenate([sig, pred, in_exc], axis=-1)

    del sig
    del pred
    del in_exc


resume_training=False
if not resume_training:
    #Training from scratch
    model.save_weights('lpcnet30_384_10_G16_00.h5')
else:
    init_epoch = 38 
    model.load_weights('lpcnet30_384_10_G16_38.h5')


# dump models to disk as we go
# checkpoint = ModelCheckpoint('lpcnet30_384_10_G16_{epoch:02d}.h5')
class MyCbk(Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        print(logs, "lr:", K.get_value(self.model.optimizer.lr), flush=True)
        self.model_to_save.save('lpcnet30_384_10_G16_%02d.h5' % epoch)
checkpoint = MyCbk(model)
sparsify = lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2), model)

parallel_model = multi_gpu_model(model, gpus=2)

# Optimizer
lr = 0.001
lr_halving = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, mode='min', min_delta=0.0001, min_lr=5e-5)
#with tf.device("/gpu:0"):
optimizer_adam = Adam(lr, amsgrad=True)
parallel_model.compile(optimizer=optimizer_adam, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#tf.get_default_graph().finalize()
parallel_model.fit([in_data, features, periods], out_exc, batch_size=batch_size, 
                   initial_epoch=init_epoch, 
                   epochs=nb_epochs, validation_split=0.08, callbacks=[checkpoint, sparsify, lr_halving])
