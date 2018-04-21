import csv
import copy
import math
import numpy as np

def load_data():
    csv_file = csv.reader(open('stat.csv','r'))
    data = []
    single_data = {}
    firstTime = True
    count = 1
    repair = True
    for user in csv_file:
        if(firstTime):
            firstTime = False
            continue
        #print(userId)
        print(count)
        count+=1
        single_data['id'] = int(user[1])
        single_data['pay_times'] = int(user[2])
        single_data['pay_point'] = []
        single_data['pay_money'] = []
        single_data['money_left'] = int(user[3])
        single_data['active_days'] = int(user[4])
        single_data['maxcannon'] = int(user[5])
        single_data['online_minutes'] = int(user[6])
        single_data['money_seq'] = []
        now = 7
        while(now < len(user) and user[now]!=''):
            if(user[now][0] == 'P'):
                length = len(single_data['pay_point'])
                single_data['pay_point'].append(now-8-length)
                single_data['pay_money'].append(int(user[now+1])-int(user[now-1]))
            else:
                single_data['money_seq'].append(int(user[now]))
            now+=1

        if(repair):
            single_data['money_seq'] += [single_data['money_seq'][-1]]*((10-len(single_data['money_seq']))%10)
        data.append(copy.deepcopy(single_data))

    return data

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

    F_max = max(seq)

    F_var = math.log10(np.array(seq).var()+0.001)

    F_mean = np.array(seq).mean()

    F_period = 0
    while(F_period < len(user['pay_point']) and user['pay_point'][F_period]<timestep-1):
        F_period += 1
    F_period += 1

    F_final = seq[-1]

    F_chargerate = F_chargemoney/(F_final+F_chargemoney+0.001)

    F_data = []
    for i in range(10):
        sliceseq = seq[int(i*timestep/10):int((i+1)*timestep/10)]
        F_data.append(sum(sliceseq)*10/timestep)

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
    alpha2 = 1
    alpha3 = 1
    alpha4 = 1

    reward = [0]*(int(len(user['money_seq'])/10))
    charge = [0] * (int(len(user['money_seq']) / 10))
    for iter in range(len(user['pay_point'])):
        charge[int(user['pay_point'][iter]/10)] += user['pay_money'][iter]
    for iter in range(len(charge)-1):
        reward[iter] = charge[iter+1]*alpha1

    if((user['active_days']>=4 and user['money_left']>10000) or (user['active_days']<4 and user['money_left']<=10000)):
        remain = 1
    else:
        remain = 0
    reward[-1] = alpha2*user['active_days'] + alpha3*user['online_minutes'] + alpha4*remain

    return reward

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
            win[iter] = 0
        else:
            win[iter] = (win[iter]-charge[iter])/temp

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

'''
#########################  test code for compute_reward  ##########################
data = load_data()
count = 1
for user in data:
    if(count == 760):
        b = 1
    reward = compute_reward(user)
    print(count)
    count+=1


#########################  test code for action_distribution  ##########################
data = load_data()
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

data = load_data()
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
