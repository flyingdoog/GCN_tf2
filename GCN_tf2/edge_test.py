from config import args
import tensorflow as tf
from sklearn.metrics import accuracy_score
from utils import *
from metrics import *
from tensorflow import keras

class edgeNet(keras.Model):
    def __init__(self, activation=tf.nn.relu, **kwargs):
        super(edgeNet, self).__init__(**kwargs)

        hidden_1 = args.hidden_1
        hidden_2 = args.hidden_2

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##

        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])

        for i in range(len(self.attentions)):
            self.nblayers.append(tf.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))
            self.selflayers.append(tf.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))

            if hidden_2>0:
                self.attentions[i].append(tf.layers.Dense(hidden_2, activation=activation , kernel_initializer = initializer))

            self.attentions[i].append(tf.layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer))

        self.attention_layers = []
        self.attention_layers.extend(self.nblayers)
        self.attention_layers.extend(self.selflayers)
        for i in range(len(self.attentions)):
            self.attention_layers.extend(self.attentions[i])

        # self.bn = tf.keras.layers.BatchNormalization(axis=-1)
    def set_fea_adj(self,fea):
        self.features = fea

    def get_attention(self, input1, input2, layer=0, training=False):

        nb_layer = self.nblayers[layer]
        selflayer = self.selflayers[layer]
        nn = self.attentions[layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout2
        else:
            dp = 1 - args.dropout2

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        # weight10 = self.bn(weight10)
        return weight10

    def call(self,inputs,training=None):
        f1_features = tf.gather(self.features,inputs[:,0])
        f2_features = tf.gather(self.features,inputs[:,1])
        return self.get_attention(f1_features,f2_features,0,training)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
all_labels = y_train + y_test+y_val
single_label = np.argmax(all_labels,axis=-1)

nodesize = features.shape[0]

# Some preprocessing
features = preprocess_features(features)
support = preprocess_adj(adj)
tuple_adj = sparse_to_tuple(adj.tocoo())


model = edgeNet()
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

features_tensor = tf.convert_to_tensor(features,dtype=tf.float32)
model.set_fea_adj(features_tensor)

pos_mask = [True if single_label[n1]==single_label[n2] else False for (n1,n2) in tuple_adj[0]]

pos_edges = sum(pos_mask)
neg_edges = len(pos_mask)-pos_edges

def check_edge(values):
    values = np.array(values).flatten()
    pos_weight = values[pos_mask]
    neg_weight = values[np.logical_not(pos_mask)]
    return np.mean(pos_weight), np.mean(neg_weight)

original_edge = np.array(tuple_adj[0])

pos_pairs = []
neg_pairs = []
nodes = np.array(range(nodesize))
train_idx = nodes[train_mask]
for node in train_idx:
    for node2 in train_idx:
        if node!=node2:
            if single_label[node] == single_label[node2]:
                pos_pairs.append([node,node2])
            else:
                neg_pairs.append([node,node2])


def sample_edge(K):
    if K>len(pos_pairs):
        K = len(pos_pairs)
    if K>len(neg_pairs):
        K = len(neg_pairs)
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    selected = pos_pairs[:K]
    labels = [1]*K
    selected.extend(neg_pairs[:int(K/2)])
    labels.extend([0] * int(K/2))
    labels = np.expand_dims(np.array(labels),-1)
    return np.concatenate((np.array(selected),labels),-1)

edge_label= sample_edge(2000)
np.random.shuffle(edge_label)
leng = edge_label.shape[0]
train_set = edge_label[:int(leng*0.8)]
test_set = edge_label[int(leng*0.8):]

for epoch in range(args.epochs):

    with tf.GradientTape() as tape:
        np.random.shuffle(train_set)
        output = model.call(train_set,training=True)
        train_labels = train_set[:,2].astype(np.float32)
        train_labels = np.expand_dims(train_labels,-1)
        edge_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=train_labels))
        grads = tape.gradient(edge_loss, model.trainable_variables)
        train_out = output.numpy().flatten()
        train_out[train_out > 0.5] = 1
        train_out[train_out <= 0.5] = 0

        print('train',accuracy_score(train_set[:,2],train_out))


    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    test_output = model.call(test_set, training=False).numpy()
    test_output[test_output>0.5]=1
    test_output[test_output <= 0.5] = 0
    test_output = test_output.astype(np.int32).flatten()
    print('test',accuracy_score(test_set[:,2],test_output))

    original_output = model.call(original_edge, training=False).numpy()
    pos,neg = check_edge(original_output)
    print('pos',pos,'neg',neg)










