import numpy as np

batch_size = 1  # Batch size for training
epochs = 100  # Number of epochs to train for
latent_dim = 256  # Latent dimensionality of the encoding space
num_samples = 1  # Number of samples to train on

from audio2numpy import open_audio
fp = "train.mp3"  # path to the mp3 file
signal, sampling_rate = open_audio(fp)

fp2 = "test.mp3"  # path to the mp3 file
signal2, sampling_rate2 = open_audio(fp2)
signal2 = np.reshape(signal, signal.shape)

length_second = len(signal) / sampling_rate
print(length_second)  # check the length of the music

##########################################3

domingo_input = signal2[:, 0]
pavarotti_input = signal[:, 0]

domingo_input = domingo_input[ -100:]
pavarotti_input = pavarotti_input[-100:]
#########################################3
domingo_input = np.reshape(domingo_input, (100, 1, 1))
pavarotti_input = np.reshape(pavarotti_input, (100, 1))


train_list = []
domingo_input2 = np.reshape(domingo_input, (100,1))
for i in range(10):
    train_list.append(domingo_input2)

gt_list = []
pavarotti_output = np.reshape(pavarotti_input, (100,))
for i in range(10):
    gt_list.append(pavarotti_output)

train_list = np.asarray(train_list)
gt_list = np.asarray(gt_list)

from keras.layers import LSTM, CuDNNLSTM
from keras.models import Sequential
from keras.layers import Dense, Flatten
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model

K.clear_session()
model = Sequential()
# Sequeatial Model  --  CuDNN LSTM
model.add(CuDNNLSTM(1, input_shape=(100,1), return_sequences=True))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))
# model.add(Dense(len(domingo_input)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)  # early stopper for training

# model.fit(train_list, gt_list, epochs=100,batch_size=1, verbose=1, callbacks=[early_stop])  # train by using 'fit'
#
# model.save('voice_transfer.h5')  # save LSTM model

model = load_model('voice_transfer.h5')
result = model.predict(train_list)  # predict with train data - overfitted

print(result)
