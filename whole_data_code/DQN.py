import numpy as np
import file_loader as fl
import preprocess as pp
import tensorflow as tf
import keras
import random
import time
import copy
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding
from keras.layers import Merge
from collections import deque


save_best_only = "score"
save = False # fake initialization

state_batch = []
next_state_batch = []
action_batch = []
reward_batch = []
y_batch = []
V_state_batch = []
V_next_state_batch = []
V_action_batch = []
V_reward_batch = []
V_y_batch = []

acc_reward_batch = []
V_acc_reward_batch = []
gamma = 0
train_size = 0
total_size = 0
state_dim = 0


class Synchronize(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.min_val_loss = 100000
        self.max_val_score = 0

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        global save_best_only
        global save
        global state_batch
        global next_state_batch
        global action_batch
        global reward_batch
        global y_batch
        global V_state_batch
        global V_next_state_batch
        global V_action_batch
        global V_reward_batch
        global V_y_batch
        global acc_reward_batch
        global V_acc_reward_batch
        global gamma
        global total_size
        global train_size
        global state_dim

        if(save):
            if(save_best_only == "loss"):
                if(self.min_val_loss > logs['val_loss']):
                    self.min_val_loss = logs['val_loss']
                    self.model.save_weights(
                        'model/' + time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + "epoch: " + str(epoch) + " val_loss" + str(logs['val_loss']))
            elif(save_best_only == "score"):
                predict_action_batch = np.argmax(self.model.predict(np.array(V_state_batch), verbose=0), 1)
                action_distribution = [0] * 14
                for item in predict_action_batch:
                    action_distribution[item] += 1

                diff_distribution = [0] * 14
                reward_sum_distribution = [0] * 14
                for iter in range(len(predict_action_batch)):
                    temp = int(abs(V_action_batch[iter] - predict_action_batch[iter]))
                    diff_distribution[temp] += 1
                    reward_sum_distribution[temp] += V_acc_reward_batch[iter]

                reward_mean_distribution = np.array(reward_sum_distribution) / (np.array(diff_distribution) + 1)
                score = 0
                for iter in range(14):
                    score += reward_mean_distribution[iter] / (iter + 1)

                print(action_distribution)
                #print(diff_distribution)
                #print(reward_mean_distribution)
                print(score)

                if(self.max_val_score < score):
                    if (self.max_val_score == 0):
                        self.max_val_score = score
                    else:
                        self.max_val_score = min(score,self.max_val_score+0.1)
                    self.model.save_weights(
                        'model/' + time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + "epoch: " + str(epoch) + " val_score" + str(score))
            else:
                self.model.save_weights('model/'+time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))+"epoch: "+str(epoch))
            #print(logs)


        y_batch = [gamma] * train_size
        Q_value_batch = np.max(self.model.predict(np.array(next_state_batch), verbose=0), 1)
        y_batch = y_batch * Q_value_batch + reward_batch

        V_y_batch = [gamma] * (total_size-train_size)
        Q_value_batch = np.max(self.model.predict(np.array(V_next_state_batch), verbose=0), 1)
        V_y_batch = V_y_batch * Q_value_batch + V_reward_batch





class DQN():
  # DQN Agent
  def __init__(self):
      # init experience replay
      self.train_replay_buffer = deque()
      self.valid_replay_buffer = []

      # init some parameters
      self.time_step = 0
      self.state_dim = 54
      self.action_dim = 14
      self.layer1_dim = 32
      self.layer2_dim = 64
      self.layer3_dim = 32
      self.training_data = []
      self.valid_data = []
      self.learning_rate = 0.00001
      self.batch_size = 32
      self.train_size = 0
      self.valid_size = 0
      self.gamma = 0.8
      self.epoch = 100
      self.dropout_rate = 0
      self.pretrain = False
      self.log_filepath = 'log/Adam20/'+time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) #/tmp/DQN_log_SGD_0.05_NoPretrain'
      self.tensorboard = True
      self.optimizer = 'adam'
      self.load_model_name = ''
      #self.save_model_name = 'pretrain'
      self.patience = 100
      self.save = True
      global save
      save = self.save

      self.create_Q_network()

      ''' test loss function
      data = np.array([[0]*self.state_dim,[0]*self.state_dim,[0]*self.state_dim,[0]*self.state_dim])
      labels = np.array([[1,2],[2,3],[9,3],[7,4]])
      self.model.fit(data, labels, epochs=10, batch_size=self.batch_size)
      '''
      self.get_data()
      #self.show_data()
      random.seed(time.time())
      self.minibatch = random.sample(self.train_replay_buffer, self.train_size)


  def get_data(self):
      self.training_data = fl.load_data(0, 400000)

      for user in self.training_data:
          statelist = pp.compute_state(user)
          actionlist = pp.compute_action(user)
          #assert(len(statelist) == len(actionlist)+1)
          rewardlist = pp.compute_reward(user)
          #assert (len(rewardlist) == len(actionlist))

          accumulate_rewardlist = copy.deepcopy(rewardlist)
          for iter in range(len(rewardlist)-2,-1,-1):
              accumulate_rewardlist[iter] += self.gamma*accumulate_rewardlist[iter+1]

          for iter in range(0,len(actionlist)):
              self.train_replay_buffer.append(copy.deepcopy([statelist[iter],statelist[iter+1],actionlist[iter],rewardlist[iter],accumulate_rewardlist[iter],user['id'],iter]))

      self.train_size = ((int)(len(self.train_replay_buffer)/self.batch_size))*self.batch_size
      print("-------------training size-------------")
      print(self.train_size)


      self.valid_data = fl.load_data(400000, 420000)

      for user in self.valid_data:
          statelist = pp.compute_state(user)
          actionlist = pp.compute_action(user)
          #assert(len(statelist) == len(actionlist)+1)
          rewardlist = pp.compute_reward(user)
          #assert (len(rewardlist) == len(actionlist))

          accumulate_rewardlist = copy.deepcopy(rewardlist)
          for iter in range(len(rewardlist)-2,-1,-1):
              accumulate_rewardlist[iter] += self.gamma*accumulate_rewardlist[iter+1]

          for iter in range(0,len(actionlist)):
              self.valid_replay_buffer.append(copy.deepcopy([statelist[iter],statelist[iter+1],actionlist[iter],rewardlist[iter],accumulate_rewardlist[iter],user['id'],iter]))

      self.valid_size = ((int)(len(self.valid_replay_buffer)/self.batch_size))*self.batch_size
      print("-------------valid size-------------")
      print(self.valid_size)
      return

  def create_Q_network(self):
      self.model = Sequential()
      self.model.add(Dense(self.layer1_dim, input_shape=(self.state_dim,)))
      if(self.dropout_rate != 0):
          self.model.add(Dropout(self.dropout_rate))
      self.model.add(Activation('sigmoid'))
      self.model.add(Dense(self.layer2_dim))
      if(self.dropout_rate != 0):
          self.model.add(Dropout(self.dropout_rate))
      self.model.add(Activation('sigmoid'))
      self.model.add(Dense(self.layer3_dim))
      if(self.dropout_rate != 0):
          self.model.add(Dropout(self.dropout_rate))
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

  def save_model(self):
      self.model.save_weights(
          'model/' + time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + "epoch: " + str(
              self.epoch) + " final")

  def metrics_test(self, filename):
      self.load_model_name = filename
      self.load_model()

      global state_batch
      global action_batch
      global acc_reward_batch
      state_batch = [data[0] for data in self.minibatch]
      action_batch = [data[2] for data in self.minibatch]
      acc_reward_batch = [data[4] for data in self.minibatch]

      predict_action_batch = self.action(state_batch)

      action_distribution = [0]*22
      for item in predict_action_batch:
          action_distribution[item] += 1

      diff_distribution = [0]*22
      reward_sum_distribution = [0]*22
      for iter in range(len(predict_action_batch)):
          temp = abs(action_batch[iter]-predict_action_batch[iter])
          diff_distribution[temp] += 1
          reward_sum_distribution[temp] += acc_reward_batch[iter]

      reward_mean_distribution = np.array(reward_sum_distribution)/(np.array(diff_distribution)+1)
      score = 0
      for iter in range(22):
          score += reward_mean_distribution[iter]/(iter+1)
      return action_distribution, diff_distribution, reward_mean_distribution, score


  def metrics_validtest_fromfile(self, filename, datafilename):
      self.load_model_name = filename
      self.load_model()

      state_batch = numpy.loadtxt(open(datafilename+"validata_state.csv", "rb"), delimiter=",", skiprows=0)
      action_batch = numpy.loadtxt(open(datafilename+"validata_action.csv", "rb"), delimiter=",", skiprows=0)
      acc_reward_batch = numpy.loadtxt(open(datafilename+"validata_reward.csv", "rb"), delimiter=",", skiprows=0)

      predict_action_batch = self.action(state_batch)

      action_distribution = [0]*22
      for item in predict_action_batch:
          action_distribution[item] += 1

      diff_distribution = [0]*22
      reward_sum_distribution = [0]*22
      for iter in range(len(predict_action_batch)):
          temp = int(abs(action_batch[iter]-predict_action_batch[iter]))
          diff_distribution[temp] += 1
          reward_sum_distribution[temp] += acc_reward_batch[iter]

      reward_mean_distribution = np.array(reward_sum_distribution)/(np.array(diff_distribution)+1)
      score = 0
      for iter in range(22):
          score += reward_mean_distribution[iter]/(iter+1)
      return action_distribution, diff_distribution, reward_mean_distribution, score


  def choose_pole(self, filename):
      self.load_model_name = filename
      self.load_model()

      global state_batch
      global action_batch
      state_batch = [data[0] for data in self.minibatch]
      action_batch = [data[2] for data in self.minibatch]
      acc_reward_batch = [data[4] for data in self.minibatch]
      id_batch = [data[5] for data in self.minibatch]
      position_batch = [data[6] for data in self.minibatch]


      predict_action_batch = self.action(state_batch)

      action_distribution = [0]*22
      for item in predict_action_batch:
          action_distribution[item] += 1

      good_list = []
      bad_list = []
      for iter in range(len(predict_action_batch)):
          temp = abs(action_batch[iter]-predict_action_batch[iter])
          if(temp<3):
              if(acc_reward_batch[iter]>2):
                  good_list.append(([id_batch[iter],position_batch[iter]],acc_reward_batch[iter]))
              elif(acc_reward_batch[iter]<0.1):
                  bad_list.append(([id_batch[iter],position_batch[iter]],acc_reward_batch[iter]))

      return good_list, bad_list


  def train_Q_network(self):

      global state_batch
      global next_state_batch
      global action_batch
      global reward_batch
      global y_batch

      global V_state_batch
      global V_next_state_batch
      global V_action_batch
      global V_reward_batch
      global V_y_batch

      global gamma
      global total_size
      global train_size
      global state_dim
      global acc_reward_batch
      global V_acc_reward_batch

      gamma = self.gamma
      train_size = self.train_size
      total_size = self.train_size+self.valid_size
      state_dim = self.state_dim

      state_batch = [data[0] for data in self.minibatch]
      next_state_batch = [data[1] for data in self.minibatch]
      action_batch = [data[2] for data in self.minibatch]
      reward_batch = [data[3] for data in self.minibatch]
      acc_reward_batch = [data[4] for data in self.minibatch]

      V_state_batch = [data[0] for data in self.valid_replay_buffer[:self.valid_size]]
      V_next_state_batch = [data[1] for data in self.valid_replay_buffer[:self.valid_size]]
      V_action_batch = [data[2] for data in self.valid_replay_buffer[:self.valid_size]]
      V_reward_batch = [data[3] for data in self.valid_replay_buffer[:self.valid_size]]
      V_acc_reward_batch = [data[4] for data in self.valid_replay_buffer[:self.valid_size]]


      y_batch = [gamma] * (self.train_size)
      Q_value_batch = np.max(self.model.predict(np.array(next_state_batch), verbose=0), 1)
      y_batch = y_batch * Q_value_batch + reward_batch

      V_y_batch = [gamma] * (self.valid_size)
      Q_value_batch = np.max(self.model.predict(np.array(V_next_state_batch), verbose=0), 1)
      V_y_batch = V_y_batch * Q_value_batch + V_reward_batch

      if(self.pretrain == True):
          for iter in range(self.train_size):
              y_batch[iter] = 0
          self.epoch = 10
          self.model.fit(np.array(state_batch), np.transpose([action_batch, y_batch]), verbose=1, epochs=self.epoch, batch_size=self.batch_size)
          self.save_model()
          return

      self.load_model()
      #print(self.evaluate())


      if(self.tensorboard):
          tb_cb = keras.callbacks.TensorBoard(log_dir=self.log_filepath, write_images=1, histogram_freq=1)
          synchro_cb = Synchronize()
          self.model.fit(np.array(state_batch),
                         np.transpose([action_batch, y_batch]),
                         validation_data=(V_state_batch, np.transpose([V_action_batch, V_y_batch])),
                         callbacks=[tb_cb, synchro_cb], verbose=2, epochs=self.epoch, batch_size=self.batch_size)
      '''
      if(self.tensorboard):
          tb_cb = keras.callbacks.TensorBoard(log_dir=self.log_filepath, write_images=1, histogram_freq=1)
          synchro_cb = Synchronize()
          es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='min')
          self.model.fit(np.array(state_batch[:self.train_size]),
                         np.transpose([action_batch[:self.train_size], y_batch[:self.train_size]]), validation_data=(
              state_batch[self.train_size:], np.transpose([action_batch[self.train_size:], y_batch[self.train_size:]])),
                         callbacks=[tb_cb, synchro_cb, es_cb], verbose=2, epochs=self.epoch, batch_size=self.batch_size)
      else:
          synchro_cb = Synchronize()
          es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='min')
          self.model.fit(np.array(state_batch[:self.train_size]),
                         np.transpose([action_batch[:self.train_size], y_batch[:self.train_size]]), validation_data=(
              state_batch[self.train_size:], np.transpose([action_batch[self.train_size:], y_batch[self.train_size:]])),
                         callbacks=[synchro_cb, es_cb], verbose=2, epochs=self.epoch, batch_size=self.batch_size)
      '''
      if (self.save):
          self.save_model()

  def show_data(self):
      f1 = open('state_data', 'w')
      f2 = open('reward_data','w')
      for item in self.train_replay_buffer:
          f1.write(str(item[0]))
          f1.write('\n')
          f2.write(str(item[3]))
          f2.write('\n')

      f1.close()
      f2.close()

  def explo_greedy_action(self,states):
      return

  def action(self,states): # no exploration, just output the action with best Q_value
      return np.argmax(self.model.predict(np.array(states), verbose=0),1)

  def action_value(self,states): # no exploration, just output highest Q_value
      return np.max(self.model.predict(np.array(states), verbose=0),1)

  def evaluate(self):
      global state_batch
      global next_state_batch
      global action_batch
      global reward_batch

      y_valid_batch = [gamma] * (self.train_size+self.valid_size)
      for iter in range(0, (self.train_size+self.valid_size)):
          if next_state_batch[iter] == 0:
              y_valid_batch[iter] = 0
              next_state_batch[iter] = [0] * state_dim
      Q_value_batch = np.max(self.model.predict(np.array(next_state_batch), verbose=0), 1)
      y_valid_batch = y_valid_batch * Q_value_batch + reward_batch

      temp2 = self.model.evaluate(np.array(state_batch[self.train_size:]),
                                  np.transpose([action_batch[self.train_size:], y_valid_batch[self.train_size:]]),
                                  batch_size=self.batch_size)
      temp1 = self.model.evaluate(np.array(state_batch[:self.train_size]),
                                  np.transpose([action_batch[:self.train_size], y_valid_batch[:self.train_size]]),
                                  batch_size=self.batch_size)

      return (temp1,temp2)