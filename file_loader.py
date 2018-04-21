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

'''
##########################  test code  ##########################

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
