from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical



####################
# 渐进式显存
####################

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.system('clear')



####################
# 载入mnist
####################

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('x_train.shape =', x_train.shape)
print('x_test.shape =', x_test.shape)
print()

print('y_train.shape =', y_train.shape)
print('y_test.shape =', y_test.shape)
print()

####################
# 模型
####################

# Build LSTM network
model = Sequential()

model.add( Bidirectional( LSTM(100, return_sequences=True), input_shape=(28, 28) ) )
# model.add( Bidirectional( LSTM(100, return_sequences=True) ) )
# model.add( LSTM(1000, return_sequences=True, input_shape=(28,28)) )

model.add( Flatten() )

model.add( Dense(256, activation='relu') )
model.add( Dropout(0.5) )
model.add( Dense(256, activation='relu') )
model.add( Dropout(0.5) )
model.add( Dense(10, activation='softmax')) 



from tensorflow.keras.optimizers import Adam

model.compile(
        optimizer = Adam(learning_rate=1e-5),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'],
        )

model.summary()



# Train
history = model.fit(
        x_train, y_train,
        epochs = 100,
        batch_size = 32,
        verbose = 1,

        max_queue_size = 320,
        workers = 80,
        use_multiprocessing = True,
        )



# Evaluate
evaluation = model.evaluate(
        x_test, y_test,
        batch_size = 32,
        verbose = 1,

        max_queue_size = 320,
        workers = 80,
        use_multiprocessing = True,
        )

print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
