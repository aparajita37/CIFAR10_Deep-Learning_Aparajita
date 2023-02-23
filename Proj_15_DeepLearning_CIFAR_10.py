import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
nc = 10 # Number of classes

#CIFAR is an acronym that stands for the Canadian 
#Institute For Advanced Research and the CIFAR-10 
#dataset was developed along with the CIFAR-100 
#dataset by researchers at the CIFAR institute.

#The dataset is comprised of 60,000 32×32 
#pixel color photographs of objects from 10 classes, 
#such as frogs, birds, cats, ships, etc. 
#The class labels and their standard associated 
#integer values are listed below.

#0: airplane
#1: automobile
#2: bird
#3: cat
#4: deer
#5: dog
#6: frog
#7: horse
#8: ship
#9: truck

(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()

#Show sample images
plt.figure(1)
imgplot1 = plt.imshow(Xtrain[nr.randint(50000)])
plt.show()

plt.figure(2)
imgplot2 = plt.imshow(Xtrain[nr.randint(50000)])
plt.show()

Xtrain = Xtrain.astype('float32')
Xtrain = Xtrain[0:20000,:] / 255.0
#we can take Xtrain[0:60000,:] also. we have simply taken a smaller subset of 20,000 images for simplification.
Xtest = Xtest.astype('float32')
Xtest = Xtest / 255.0

ytrain = ytrain[:,0]
ytrainEnc = tf.one_hot(ytrain[0:20000], depth=nc)

ytest = ytest[:,0]
ytestEnc = tf.one_hot(ytest, depth=nc)

model=Sequential()
model.add(Conv2D(32, (3,3), activation="relu", kernel_initializer='he_uniform', padding= 'same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), activation="relu", kernel_initializer='he_uniform', padding= 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation="relu", kernel_initializer='he_uniform', padding= 'same'))
model.add(Conv2D(64, (3,3), activation="relu", kernel_initializer='he_uniform', padding= 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation="relu", kernel_initializer='he_uniform', padding= 'same'))
model.add(Conv2D(128, (3,3), activation="relu", kernel_initializer='he_uniform', padding= 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu",kernel_initializer='he_uniform' ))
model.add(Dense(10,activation="softmax"))

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(Xtrain, ytrainEnc, epochs=20, batch_size=100,validation_data=(Xtest, ytestEnc))
#increase the size of epochs and decrease the size of batch for more accuracy. We can also add more layers for more accuracy

ypred = model.predict(Xtest)
ypred = np.argmax(ypred,axis=1)

score = accuracy_score(ypred,ytest)
print('Accuracy score is',100*score,'%')