import numpy as np
import file_loader as fl
import preprocess as pp
import tensorflow as tf
import keras
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Embedding
from keras.layers import Merge
from collections import deque

class DQN():
  # DQN Agent
  def __init__(self):
      # init experience replay
      self.replay_buffer = deque()

      # init some parameters
      self.time_step = 0
      self.state_dim = 23
      self.action_dim = 22
      self.layer1_dim = 32
      self.layer2_dim = 32
      self.data = []
      self.learning_rate = 0.00005
      self.batch_size = 32
      self.train_size = 32000
      self.gamma = 0.95
      self.epoch = 10000
      self.pretrain = True
      self.log_filepath = 'log/pretrain'#/tmp/DQN_log_SGD_0.05_NoPretrain'
      self.tensorboard = True
      self.optimizer = 'adam'
      self.load_model_name = ''
      self.save = True

      self.create_Q_network()

      ''' test loss function
      data = np.array([[0]*self.state_dim,[0]*self.state_dim,[0]*self.state_dim,[0]*self.state_dim])
      labels = np.array([[1,2],[2,3],[9,3],[7,4]])
      self.model.fit(data, labels, epochs=10, batch_size=self.batch_size)
      '''

      self.get_data()


  def get_data(self):
      self.data = fl.load_data()
      for user in self.data:
          actionlist = pp.compute_action(user)
          rewardlist = pp.compute_reward(user)
          statelist = []
          length = len(user['money_seq'])
          for timestep in range(10, length + 1, 10):
              statelist.append(pp.abstract_feature(user, timestep))

          # assert (len(actionlist) == len(rewardlist) and len(actionlist) == len(statelist))
          statelist.append(0)
          for iter in range(0,len(actionlist)):
              self.replay_buffer.append([statelist[iter],statelist[iter+1],actionlist[iter],rewardlist[iter]])

      pp.rewardNormalization(self.replay_buffer)
      #print(len(self.replay_buffer))
      return

  def create_Q_network(self):
      self.model = Sequential()
      self.model.add(Dense(self.layer1_dim, input_shape=(self.state_dim,)))
      self.model.add(Activation('sigmoid'))
      self.model.add(Dense(self.layer2_dim))
      self.model.add(Activation('sigmoid'))
      self.model.add(Dense(self.action_dim))

      if(self.optimizer == 'adam'):
          myOptimizer = keras.optimizers.Adam(lr=self.learning_rate)
      elif(self.optimizer == 'sgd'):
          myOptimizer = keras.optimizers.SGD(lr=self.learning_rate, momentum=0., decay=0., nesterov=False)
      self.model.compile(loss=[self.my_loss_action], optimizer=myOptimizer)
      self.model.summary()


  def my_loss_action(self, y_true, y_pred):
      y_true = tf.transpose(y_true)
      action = y_true[0]
      Q_true = y_true[1]
      action = tf.cast(tf.expand_dims(action, 1),tf.int32)
      index = tf.expand_dims(tf.range(0, self.batch_size), 1)
      concated = tf.concat([index, action], 1)
      onehot_action = tf.sparse_to_dense(concated, [self.batch_size, self.action_dim], 1.0, 0.0)
      Q_value = tf.reduce_sum(y_pred*onehot_action,1)
      return tf.reduce_mean(tf.square(Q_true - Q_value))

  def load_model(self):
      if(self.load_model_name != ''):
          self.model.load_weights('model/'+ self.load_model_name)

  def save_model(self,name):
      self.model.save_weights('model/'+ name)

  def train_Q_network(self):
      #while(1):
      minibatch = random.sample(self.replay_buffer, self.train_size)
      state_batch = [data[0] for data in minibatch]
      next_state_batch = [data[1] for data in minibatch]
      action_batch = [data[2] for data in minibatch]
      reward_batch = [data[3] for data in minibatch]

      y_batch = [self.gamma]*self.train_size
      for iter in range(0, self.train_size):
          if minibatch[iter][1] == 0:
              y_batch[iter] = 0
              next_state_batch[iter] = [0]*self.state_dim

      Q_value_batch = self.action_value(next_state_batch)
      y_batch = y_batch*Q_value_batch+reward_batch
      if(self.pretrain == True):
          for iter in range(self.train_size):
              y_batch[iter] = 0
          self.epoch = 200

      self.load_model()
      if(self.tensorboard):
          tb_cb = keras.callbacks.TensorBoard(log_dir=self.log_filepath, write_images=1, histogram_freq=1)
          self.model.fit(np.array(state_batch), np.transpose([action_batch,y_batch]),callbacks=[tb_cb], verbose=2,epochs=self.epoch, batch_size=self.batch_size)
      else:
          self.model.fit(np.array(state_batch), np.transpose([action_batch,y_batch]), verbose=2,epochs=self.epoch, batch_size=self.batch_size)

      if(self.save):
          self.save_model("pretrain")


  def show_data(self):
      f1 = open('state_data', 'a')
      f2 = open('reward_data','a')
      for item in self.replay_buffer:
          f1.write(str(item[0]))
          f1.write('\n')
          f2.write(str(item[3]))
          f2.write('\n')

  def explo_greedy_action(self,states):
      return

  def action(self,states): # no exploration, just output the action with best Q_value
      return np.argmax(self.model.predict(np.array(states), verbose=0),1)

  def action_value(self,states): # no exploration, just output highest Q_value
      return np.max(self.model.predict(np.array(states), verbose=0),1)