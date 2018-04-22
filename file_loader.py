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

