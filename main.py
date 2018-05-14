from DQN import DQN



agent = DQN()
#agent.show_data()
#agent.train_Q_network()
#action_distribution, diff_distribution, reward_mean_distribution,score = agent.metrics_test('20180507204249epoch: 100 final')
#print(action_distribution)
#print(diff_distribution)
#print(reward_mean_distribution)
#print(score)

good_list, bad_list = agent.choose_pole('20180507204249epoch: 100 final')
print("good_list")
for item in good_list:
    print(item)
print("bad_list")
for item in bad_list:
    print(item)