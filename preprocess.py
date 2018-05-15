import csv
import copy
import math
import numpy as np
import file_loader as fl

def abstract_feature(user,timestep):
    seq = user['money_seq'][:timestep]

    F_step = timestep/10
    F_chargetime = 0
    F_chargemoney = 0
    for iter in range(len(user['pay_point'])):
        if(user['pay_point'][iter] < timestep):
            F_chargetime += 1
            F_chargemoney += user['pay_money'][iter]

    F_uptime = 0
    F_contUptime = 0
    uptemp = 0
    F_downtime = 0
    F_contDowntime = 0
    downtemp = 0
    fromnum = 0
    addflag = 0
    if(user['pay_point'] == [] or user['pay_point'][-1]<timestep-1):
        user['pay_point'].append(timestep-1)
        addflag = 1
    for item in user['pay_point']:
        if(item >= timestep):
            item = timestep-1
        for iter in range(fromnum,item):
            if(seq[iter] < seq[iter+1]):
                F_uptime+=1
                uptemp+=1
                downtemp = 0
                if(uptemp > F_contUptime):
                    F_contUptime = uptemp
            elif(seq[iter] > seq[iter+1]):
                F_downtime+=1
                downtemp+=1
                uptemp = 0
                if(downtemp > F_contDowntime):
                    F_contDowntime = downtemp
        fromnum = item+1
        if(fromnum >= timestep):
            break
    if(addflag == 1):
        user['pay_point'].pop()
    F_uptime /= F_step
    F_downtime /= F_step

    F_max = math.log10(max(seq)+1.0)

    if(min(seq)<0):
        wer = 1

    F_var = np.log(np.array(seq)+1.0).var()

    F_mean = np.log(np.array(seq)+1.0).mean()

    F_period = F_step / (F_chargetime+1)

    F_final = seq[-1]

    F_chargerate = F_chargemoney/(F_final+F_chargemoney+0.001)

    F_final = math.log10(F_final+1.0)
    F_chargemoney = math.log10(F_chargemoney+1.0)

    F_data = []
    for i in range(10):
        sliceseq = seq[int(i*timestep/10):int((i+1)*timestep/10)]
        F_data.append(math.log10(sum(sliceseq)*10/timestep+1.0))

    F_step /= 5

    feature = [
                  F_step,
                  F_chargetime, F_chargemoney, F_chargerate,
                  F_uptime, F_contUptime, F_downtime, F_contDowntime,
                  F_max, F_var, F_mean,
                  F_period,
                  F_final,
              ] + F_data

    return feature

def compute_reward(user):
    alpha1 = 1
    alpha2 = 0.5
    alpha3 = 0.0002
    alpha4 = 0.5

    reward = [0]*(int(len(user['money_seq'])/10))
    charge = [0] * (int(len(user['money_seq']) / 10))
    for iter in range(len(user['pay_point'])):
        charge[int(user['pay_point'][iter]/10)] += user['pay_money'][iter]
    for iter in range(len(charge)-1):
        reward[iter] = math.log10(charge[iter+1]+1.0)*alpha1

    if((user['active_days']>=4 and user['money_left']>10000) or (user['active_days']<4 and user['money_left']<=10000)):
        remain = 1
    else:
        remain = 0
    reward[-1] = alpha2*user['active_days'] + alpha3*user['online_minutes'] + alpha4*remain

    return reward

# check point: -1 -0.9 -0.8 ... 0.9 1
def convert_action(num):
    if(num < -1):
        return 0
    elif(num > 1):
        return 21
    else:
        return int((num+1.1)*10)

def compute_action(user):
    charge = [0]*(int(len(user['money_seq'])/10))
    win = [0]*(int(len(user['money_seq']) / 10))

    for iter in range(len(user['pay_point'])):
        charge[int(user['pay_point'][iter]/10)] += user['pay_money'][iter]

    win[0] = user['money_seq'][9]
    for iter in range(9,len(user['money_seq'])-1,10):
        win[int((iter+1)/10)] = user['money_seq'][iter+10] - user['money_seq'][iter]

    for iter in range(0,len(win)):
        temp = max(user['money_seq'][10 * iter:10 * iter + 10])
        if(temp == 0):
            win[iter] = convert_action(0)
        else:
            win[iter] = convert_action((win[iter]-charge[iter])/temp)

    return win

def action_distribution(data):
    #### test maxnum and minnum ####
    maxnum = 0
    minnum = 0
    for user in data:
        temp = compute_action(user)
        tempmax = max(temp)
        tempmin = min(temp)
        if(tempmax>1):
            print(tempmax)
        if(tempmin<-10):
            print(tempmin)
        maxnum = max(maxnum,tempmax)
        minnum = min(minnum,tempmin)
        if(max(temp)>1):
            print()
    print(maxnum)
    print(minnum)

    #### test macro distribution ####
    dist = [0]*22
    for user in data:
        temp = compute_action(user)
        for item in temp:
            if(item<-20):
                item = -20
            dist[int(item)+20]+=1
    print(dist)

    #### test micro distribution ####
    dist = [0]*31
    for user in data:
        temp = compute_action(user)
        for item in temp:
            if(item>=-1):
                dist[int((item+1)*10)]+=1
    print(dist)
    return 1

def rewardNormalization(data):
    #count = 0
    #allreward = []
    #for item in data:
    #    allreward.append(item[3])
    #    if(item[3]>20):
    #        count += 1

    #mean = np.mean(allreward)
    #max = np.max(allreward)
    for iter in range(len(data)):
        #data[iter][3] = (data[iter][3]-mean)/max
        data[iter][3] = data[iter][3] / 20
        data[iter][4] = data[iter][4] / 20

'''
#########################  test code for compute_reward  ##########################
data = fl.load_data()
count = 1
for user in data:
    if(count == 760):
        b = 1
    reward = compute_reward(user)
    print(count)
    count+=1


#########################  test code for action_distribution  ##########################
data = fl.load_data()
action_distribution(data)


#########################  test code for compute_action  ##########################
user = {}
user['money_seq'] = [2,5,10,1,200,400,100,50,60,600,1000,3000,500,15000,10000,6000,4000,6000,5000,200]
user['pay_point'] = [3,10,12]
user['pay_money'] = [200,3000,14000]
action = compute_action(user)


#########################  test code for abstract_feature  ##########################
user = {}
user['money_seq'] = [2,5,10,1,200,400,100,50,60,600,1000,3000]
user['pay_point'] = [3,10]
user['pay_money'] = [200,3000]
abstract_feature(user,10)

data = fl.load_data()
count = 1
for user in data:
    if(count == 84):
        b = 1
    length = len(user['money_seq'])
    for timestep in range(10,length+1,10):
        a = abstract_feature(user,timestep)
    print(count)
    count+=1
'''
