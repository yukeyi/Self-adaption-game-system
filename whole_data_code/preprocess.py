import csv
import copy
import math
import numpy as np
import file_loader as fl

def abstract_feature(user, timestep, blanktimestep):

    F_step = timestep/3
    F_blankstep = blanktimestep

    payM = [0]*timestep
    payT = [0]*timestep
    payR = [0]
    for iter in range(timestep):
        if iter in user:
            if(user[iter]['pay_money'] != 0):
                payR.append(user[iter]['pay_money'])
                payM[iter] += user[iter]['pay_money']
                payT[iter] = 1
    F_chargetime = sum(payT)*10/F_step
    F_chargemoney = sum(payM)/(sum(payT)+0.1)
    F_chargemax = max(payM)/5
    F_chargevar = np.std(payR)/10
    F_chargetimeTrend = [0, 0, 0]
    F_chargemoneyTrend = [0, 0, 0]
    for iter in range(3):
        F_chargetimeTrend[iter] = sum(payT[int(iter*timestep/3):int((iter+1)*timestep/3)])
        F_chargemoneyTrend[iter] = sum(payM[int(iter*timestep/3):int((iter+1)*timestep/3)])

    temp1 = (sum(F_chargetimeTrend)+0.1)
    temp2 = (sum(F_chargemoneyTrend)+0.1)
    for iter in range(3):
        F_chargetimeTrend[iter] /= temp1
        F_chargetimeTrend[iter] *= 10
        F_chargemoneyTrend[iter] /= temp2
        F_chargemoneyTrend[iter] *= 10

    gameD = [0]*timestep
    gameT = [0]*timestep
    gameR = []
    for iter in range(timestep):
        if iter in user:
            gameT[iter] += np.log(user[iter]['online_time']+1)
            gameR.append(np.log(user[iter]['online_time']+1))
            gameD[iter] = 1
    F_gameday = sum(gameD)*10/F_step
    F_gameavertime = sum(gameT)/sum(gameD)
    F_gamevar = np.std(gameR)
    F_gamemax = max(gameR)
    F_shortgameTime = 0
    F_longgameTime = 0
    for item in gameR:
        #print(item)
        if(item < 2):
            F_shortgameTime += 1
        elif(item > 5):
            F_longgameTime += 1
    F_lastgame = gameT[-3:]
    F_gametrend = [0,0,0]
    for iter in range(3):
        F_gametrend[iter] = sum(gameT[int(iter*timestep/3):int((iter+1)*timestep/3)])

    temp = (sum(F_gametrend)+0.1)
    for iter in range(3):
        F_gametrend[iter] /= temp
        F_gametrend[iter] *= 10

    vipD = 0
    vipR = []
    diamondR = []
    cannonR = []
    for iter in range(timestep):
        if iter in user:
            vipR.append(user[iter]['vip'])
            diamondR.append(user[iter]['diamond'])
            cannonR.append(user[iter]['max_cannon'])
            vipD += 1
    F_vipLast = np.sqrt(vipR[-1])
    F_vipaver = np.sqrt(sum(vipR)/vipD)
    F_vipvar = np.sqrt(np.std(vipR))
    F_vipmax = np.sqrt(max(vipR))
    F_diamondLast = np.sqrt(diamondR[-1])
    F_diamondaver = np.sqrt(sum(diamondR)/vipD)
    F_diamondvar = np.sqrt(np.std(diamondR))
    F_diamondmax = np.sqrt(max(diamondR))
    F_cannonLast = np.sqrt(cannonR[-1])
    F_cannonaver = np.sqrt(sum(cannonR)/vipD)
    F_cannonvar = np.sqrt(np.std(cannonR))
    F_cannonmax = np.sqrt(max(cannonR))


    uptime = 0
    totaltime = 0
    F_contDowntime = 0
    F_contUptime = 0
    F_goldMax = 0
    gold_data = []
    for iter in range(timestep):
        if iter in user:
            gold_data += user[iter]['gold_seq']
            uptemp = 0
            downtemp = 0
            for iter2 in range(len(user[iter]['gold_seq'])-1):
                if(user[iter]['gold_seq'][iter2]>F_goldMax):
                    F_goldMax = user[iter]['gold_seq'][iter2]
                if(user[iter]['gold_seq'][iter2] < user[iter]['gold_seq'][iter2+1]):
                    uptemp += 1
                    uptime += 1
                    if(downtemp > F_contDowntime):
                        F_contDowntime = downtemp
                    downtemp = 0
                else:
                    downtemp += 1
                    if(uptemp > F_contUptime):
                        F_contUptime = uptemp
                    uptemp = 0

            totaltime += len(user[iter]['gold_seq'])-1

    F_uprate = uptime/(totaltime+1) * 10

    for iter in range(timestep-1,-1,-1):
        if iter in user:
            F_goldLast = user[iter]['gold_seq'][-1]
            F_goldFirst = user[iter]['gold_seq'][0]
            break

    gold_leftR = []
    for iter in range(timestep):
        if iter in user:
            gold_leftR.append(np.log(user[iter]['gold_left']+1.0))
    F_gold_left_aver = sum(gold_leftR)/vipD
    F_gold_left_var = np.std(gold_leftR)
    F_goldtrend = [0]*10
    for iter in range(10):
        F_goldtrend[iter] = sum(gold_data[int(iter*len(gold_data)/10):int((iter+1)*len(gold_data)/10)])
    temp = (sum(F_goldtrend)+0.1)
    for iter in range(10):
        F_goldtrend[iter] /= temp
        F_goldtrend[iter] *= 30


    feature = [
                  F_step, F_blankstep,
                  F_chargetime, F_chargemoney, F_chargemax, F_chargevar] + F_chargetimeTrend + F_chargemoneyTrend + [
                  F_gameday, F_gameavertime, F_gamevar, F_gamemax, F_shortgameTime, F_longgameTime] + F_lastgame + F_gametrend + [
                  F_vipLast, F_vipaver, F_vipvar, F_vipmax,
                  F_diamondLast, F_diamondaver, F_diamondvar, F_diamondmax,
                  F_cannonLast, F_cannonaver, F_cannonvar, F_cannonmax,
                  F_uprate, F_contDowntime, F_contUptime, F_goldMax, F_goldLast, F_goldFirst, F_gold_left_aver, F_gold_left_var
              ] + F_goldtrend

    return feature



def compute_state(user):
    statelist = []

    online = []
    for iter in range(30):
        if iter in user:
            online.append(int(iter/3))
    online = list(set(online))

    for iter in range(len(online)):
        if(iter == len(online)-1):
            statelist.append(abstract_feature(user, online[iter] * 3 + 3,0))
        else:
            statelist.append(abstract_feature(user, online[iter] * 3 + 3,3*(online[iter+1]-online[iter])))

    #isfirst = 1
    #for iter in range(10):
    #    if(online[iter] == 1):
    #        statelist.append(abstract_feature(user, iter * 3 + 3,))
    #        if(isfirst == 1):
    #            isfirst = 0
    #        else:
    #            statelist.append(abstract_feature(user,iter*3+3))

    return statelist

def compute_reward(user):
    alpha1 = 1
    alpha2 = 2
    alpha3 = 3
    alpha4 = 0.5

    rewardlist = []

    online = []
    for iter in range(30):
        if iter in user:
            online.append(int(iter/3))
    online = list(set(online))

    for iter in range(1,len(online)):
        item = online[iter]
        activedays = 0
        activetime = 0
        chargemoney = 0
        lastactivetime = 0
        for iter2 in range(3 * item, 3 * item + 3):
            if iter2 in user:
                activedays += 1
                activetime += user[iter2]['online_time']
                chargemoney += user[iter2]['pay_money']

        activedays = activedays*activedays
        activetime = np.log(activetime+1.0)
        if (activetime > lastactivetime):
            lastactivetime = activetime
            activetime*=1.2
        else:
            lastactivetime = activetime
            activetime*=0.8
        chargemoney = np.log(chargemoney+1.0)

        rewardlist.append(alpha1*activedays + alpha2*activetime + alpha3*chargemoney)
        #print([alpha1*activedays,alpha2*activetime,alpha3*chargemoney])

        if(iter == len(online)-1):
            #print(rewardlist)
            if(np.max(rewardlist) == rewardlist[-1]):
                futurereward = sum(rewardlist)
            else:
                x1 = np.argmax(rewardlist)
                y1 = rewardlist[x1]
                x2 = len(rewardlist)-1
                y2 = rewardlist[x2]
                x3 = (y1*x2-y2*x1)/(y1-y2+0.1)
                futurereward = (x3-x2)*y2/2
                #print(futurereward)
            assert (futurereward >= 0)
            rewardlist[-1] += alpha4 * futurereward
            #print(alpha4 * futurereward)

    # rewardNormalization
    for iter in range(len(rewardlist)):
        rewardlist[iter] /= 100
        if(rewardlist[iter]>2):
            rewardlist[iter] = 2

    #print(rewardlist)
    return rewardlist


# check point: -1 -0.9 -0.8 ... 0.9 1
#count = 0
#distrib = [0]*14
def convert_action(num, temp):
    #global count
    #if(num < -20):
    #    count+=1
    #    print(count)
    if(num < -27):
        add = 0
    elif(num > 1):
        add = 29
    else:
        add = int(num+28)
    if(temp > 0):
        add += 30
    else:
        add = 29-add

    add = int(add/2)
    if(add >=27):
        add = 13
    elif(add >= 21):
        add = 12
    elif(add >= 11):
        add = 11

    #distrib[add]+=1
    #print(distrib)
    return add

def compute_action(user):
    actionlist = []

    online = []
    for iter in range(30):
        if iter in user:
            online.append(int(iter/3))
    online = list(set(online))

    for iter in range(len(online)-1):
        #action = f(user, online[iter], online[iter+1])
        maxgold_before = 0
        gold_pay = 0
        gold_data = []

        for iter3 in range(3*online[iter]+3):
            if iter3 in user:
                for iter2 in range(len(user[iter3]['gold_seq']) - 1):
                    if (user[iter3]['gold_seq'][iter2] > maxgold_before):
                        maxgold_before = user[iter3]['gold_seq'][iter2]

        for iter3 in range(3 * online[iter+1], 3 * online[iter+1]+3):
            if iter3 in user:
                gold_data += user[iter3]['gold_seq']
                gold_pay += user[iter3]['gold_pay']

        temp = (np.exp(gold_data[-1])-np.exp(gold_data[0])-gold_pay)
        action = np.log((abs(temp)+1) / (np.exp(maxgold_before)+1))
        #print(action)
        actionlist.append(convert_action(action, temp))

    return actionlist


'''
def action_distribution(data):
    #### test maxnum and minnum ####
    maxnum = 0
    minnum = 0
    for user in data:
        temp = compute_action(user)
        if(temp != []):
            tempmax = max(temp)
            tempmin = min(temp)
            if(tempmax>0):
                print(tempmax)
            #if(tempmin<-10):
            #    print(tempmin)
            maxnum = max(maxnum,tempmax)
            minnum = min(minnum,tempmin)
    print(maxnum)
    print(minnum)
    return
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
