import csv
import copy
import math
import numpy as np

def convert(st):
    if(st[0] == 'p'):
        return -1
    else:
        return int(st)

def load_data(fromnum, tonum):
    data = []
    count = 0
    for day in range(30):
        f = open('3-'+str(day+1)+'.log', 'r')
        single_day_data = f.readlines()
        for iter in range(0,len(single_day_data)):
            temp1 = single_day_data[iter].split(';')
            temp2 = temp1[0].split(',') + temp1[1].split(',')
            temp2 = [convert(st) for st in temp2[:-1]]
            gold_pay = 0
            for i in range(len(temp2)):
                if(temp2[i] < 0):
                    if(i != 0):
                        temp2[i] = temp2[i-1]
                    else:
                        temp2[i] = 0

                    if(i != 0 and i != len(temp2)-1):
                        gold_pay += temp2[i+1]-temp2[i-1]
            #print(temp2[0])
            if(temp2[0]>tonum or temp2[0]<fromnum):
                continue
            find = 0
            for person in data:
                if(person['id'] == temp2[0]):
                    find = 1
                    person_oneday = {}
                    person_oneday['vip'] = temp2[2]
                    person_oneday['diamond'] = temp2[3]
                    person_oneday['pay_money'] = temp2[4]
                    person_oneday['gold_left'] = temp2[5]
                    person_oneday['max_cannon'] = temp2[6]
                    person_oneday['online_time'] = (temp2[-2] - temp2[7]) / 30000
                    person_oneday['gold_seq'] = []
                    person_oneday['gold_pay'] = gold_pay
                    for iter2 in range(8, len(temp2), 2):
                        person_oneday['gold_seq'].append(np.log(temp2[iter2]+1.0))
                    person[day] = copy.deepcopy(person_oneday)
                    break

            if(find == 0):
                count+=1
                #print(count)
                person = {}
                person['id'] = temp2[0]
                person_oneday = {}
                person_oneday['vip'] = temp2[2]
                person_oneday['diamond'] = temp2[3]
                person_oneday['pay_money'] = temp2[4]
                person_oneday['gold_left'] = temp2[5]
                person_oneday['max_cannon'] = temp2[6]
                person_oneday['online_time'] = (temp2[-2]-temp2[7])/30000
                person_oneday['gold_seq'] = []
                person_oneday['gold_pay'] = gold_pay
                for iter2 in range(8, len(temp2),2):
                    person_oneday['gold_seq'].append(np.log(temp2[iter2] + 1.0))
                person[day] = person_oneday
                data.append(copy.deepcopy(person))

        print(day)
        f.close()

    return data

def make_small_dataset():
    for day in range(30):
        f1 = open('3-'+str(day+1)+'.log', 'r')
        f2 = open('3-'+str(day+1)+'small.log', 'w')
        while(1):
            data = f1.readline()
            if(int(data.split(',')[0])<200000):
                f2.write(data)
            else:
                f1.close()
                f2.close()
                break

#make_small_dataset()