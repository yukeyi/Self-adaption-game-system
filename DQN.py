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
      self.learning_rate = 0.001
      self.batch_size = 4
      self.gamma = 0.8
      self.miniepo = 10

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

      #print(len(self.replay_buffer))
      return

  def create_Q_network(self):
      self.model = Sequential()
      self.model.add(Dense(self.layer1_dim, input_shape=(self.state_dim,)))
      self.model.add(Activation('sigmoid'))
      self.model.add(Dense(self.layer2_dim))
      self.model.add(Activation('sigmoid'))
      self.model.add(Dense(self.action_dim))
      self.model.add(Activation('sigmoid'))

      myOptimizer = keras.optimizers.Adam(lr=self.learning_rate)
      self.model.compile(loss=[self.my_loss_action], optimizer=myOptimizer)


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

  def train_Q_network(self):
      while(1):
          minibatch = random.sample(self.replay_buffer, self.batch_size)
          state_batch = [data[0] for data in minibatch]
          next_state_batch = [data[1] for data in minibatch]
          action_batch = [data[2] for data in minibatch]
          reward_batch = [data[3] for data in minibatch]

          y_batch = [self.gamma]*self.batch_size
          for iter in range(0, self.batch_size):
              if minibatch[iter][1] == 0:
                  y_batch[iter] = 0
                  next_state_batch[iter] = [0]*self.state_dim

          Q_value_batch = self.action_value(next_state_batch)
          y_batch = y_batch*Q_value_batch+reward_batch

          # np.array([[1,2],[2,3],[9,3],[7,4]])
          self.model.fit(np.array(state_batch), np.transpose([action_batch,y_batch]), epochs=self.miniepo, batch_size=self.batch_size)

  def explo_greedy_action(self,states):
      return

  def action(self,states): # no exploration, just output the action with best Q_value
      return np.argmax(self.model.predict(np.array(states), verbose=0),1)

  def action_value(self,states): # no exploration, just output highest Q_value
      return np.max(self.model.predict(np.array(states), verbose=0),1)