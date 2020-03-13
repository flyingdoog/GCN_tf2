from config import *
from layers import *
from metrics import *
from tensorflow import keras

class GCN(keras.Model):
    def __init__(self, input_dim, output_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        self.layers_ = []

        layer0 = GraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0],
                                  activation=tf.nn.relu)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_],
                                      activation=tf.nn.relu)
            self.layers_.append(layertemp)

        layer_1 = GraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim,
                                            activation=lambda x: x)
        self.layers_.append(layer_1)
        self.hiddens = hiddens

    def call(self,inputs,training=None):
        x, support  = inputs
        for layer in self.layers_:
            x = layer.call((x,support),training)
        return x

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
