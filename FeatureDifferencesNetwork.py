from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import *


class FeatureDifferencesNetwork():
    def addUntrainableImageConvLayers(self,  inputOne, inputTwo, minLayer, maxExclusiveLayer):
        imageConvBase = VGG16(include_top=False)
        outputLayers = [inputOne, inputTwo]
        for i in range(maxExclusiveLayer - minLayer):
            newLayer = imageConvBase.layers[i + minLayer]
            newLayer.trainable = False
            newLayer.name = newLayer.name + newLayer.name # all layers names must be unique but we use this model twice, this is temporary
            outputLayers[0] = newLayer(outputLayers[0])
            outputLayers[1] = newLayer(outputLayers[1])

        subtracted = Subtract()
        subtracted.trainable = False
        subtracted = subtracted(outputLayers)

        #  TODO: We may need to normalize data or something for tanh to give proper values?
        activation = Activation('tanh')
        activation.trainable = False

        return activation(subtracted)

    def __init__(self):
        self.inputs = []
        self.inputs.append(Input(shape=(640, 640, 3)))
        self.inputs.append(Input(shape=(640, 640, 3)))

        imageConvLayers = self.addUntrainableImageConvLayers(self.inputs[0], self.inputs[1], 1, 18)

        upsampleLayer = UpSampling2D(size=16)(imageConvLayers) #fixme, size use to be 4 but the network breaks if the shape isn't exact
        translatorLayer = Convolution2D(filters=3, kernel_size=1, activation="relu")(upsampleLayer)


        self.network = translatorLayer # TODO: Change this whenever you add a layer

        # output = translatorLayer  # TODO: Change this whenever you add a layer
        #
        # self.net = Model([inputOne, inputTwo], output)
        # self.net.compile(optimizer="adam", loss="mse")

