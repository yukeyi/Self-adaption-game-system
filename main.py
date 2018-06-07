from DQN import DQN



agent = DQN()
'''decide action distribution'''
#agent.show_data()

'''training network'''
agent.train_Q_network()

'''get test score on whole dataset or only validation dataset'''
'''
filename_list = ['',
                 '20180515015900epoch: 0 val_loss0.0304607906875',
                 '20180515015903epoch: 1 val_loss0.0270299295048',
                 '20180515015905epoch: 2 val_loss0.0257106056235',
                 '20180515015907epoch: 3 val_loss0.0251836379201',
                 '20180515015909epoch: 4 val_loss0.0248966536134',
                 '20180515015913epoch: 6 val_loss0.0234001754869',
                 '20180515015915epoch: 7 val_loss0.0227902077726',
                 '20180515015917epoch: 8 val_loss0.0225628141319',
                 '20180515015919epoch: 9 val_loss0.0221484689282',
                 '20180515015921epoch: 10 val_loss0.0218110977887',
                 '20180515015928epoch: 13 val_loss0.0216134398457',
                 '20180515015932epoch: 15 val_loss0.0204433792553',
                 '20180515015941epoch: 19 val_loss0.0194071854627',
                 '20180515015951epoch: 24 val_loss0.0188520855045',
                 '20180515020000epoch: 28 val_loss0.018557097111',
                 '20180515020004epoch: 30 val_loss0.0180960219708',
                 '20180515020015epoch: 35 val_loss0.0177139892501',
                 '20180515020025epoch: 40 val_loss0.0175061543091',
                 '20180515020027epoch: 41 val_loss0.0169781016593',
                 '20180515020039epoch: 47 val_loss0.0167688590106',
                 '20180515020047epoch: 51 val_loss0.0165735199017',
                 '20180515020101epoch: 58 val_loss0.0163849760291',
                 '20180515020115epoch: 65 val_loss0.0159745575695',
                 '20180515020125epoch: 70 val_loss0.0157163927829',
                 '20180515020129epoch: 72 val_loss0.0156728314382',
                 '20180515020136epoch: 75 val_loss0.0154680465588',
                 '20180515020138epoch: 76 val_loss0.0154129659829',
                 '20180515020154epoch: 83 val_loss0.0152915993464',
                 '20180515020156epoch: 84 val_loss0.0151340753159',
                 '20180515020158epoch: 85 val_loss0.0149536524521',
                 '20180515020229epoch: 100 val_loss0.0146620608279',
                 '20180515020314epoch: 122 val_loss0.014519978303',
                 '20180515020323epoch: 127 val_loss0.0144781638126',
                 '20180515020345epoch: 138 val_loss0.0143279296063',
                 '20180515020501epoch: 176 val_loss0.0143072061725',
                 '20180515020531epoch: 191 val_loss0.0140770534207',
                 '20180515020700epoch: 235 val_loss0.0139068905965',
                 '20180515020827epoch: 279 val_loss0.0136183766532',
                 '20180515020849epoch: 290 val_loss0.0135558255429',
                 '20180515020919epoch: 305 val_loss0.0133253401508',
                 '20180515021021epoch: 336 val_loss0.0132035456686',
                 '20180515021139epoch: 372 val_loss0.0131515876163',
                 '20180515021211epoch: 387 val_loss0.0131155685486',
                 '20180515021327epoch: 424 val_loss0.0129447572971',
                 '20180515021432epoch: 455 val_loss0.0128705340731',
                 '20180515021616epoch: 507 val_loss0.0128048932329',
                 '20180515021644epoch: 520 val_loss0.012787099873',
                 '20180515021923epoch: 602 val_loss0.012773459511',
                 '20180515022419epoch: 752 val_loss0.0127613343293',
                 '20180515023019epoch: 929 val_loss0.0127312277207',
                 'pretrain']

for item in filename_list:
    action_distribution, diff_distribution, reward_mean_distribution,score = agent.metrics_validtest_fromfile(item,'20180515015855')
    print(item)
    print(action_distribution)
    print(diff_distribution)
    print(reward_mean_distribution)
    print(score)
    print('\n')
'''

'''choose good and bad example'''
#good_list, bad_list = agent.choose_pole('20180507204249epoch: 100 final')
#print("good_list")
#for item in good_list:
#    print(item)
#print("bad_list")
#for item in bad_list:
#    print(item)