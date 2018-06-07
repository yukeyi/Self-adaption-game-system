from pylab import *

font = {
        'family' : 'times new roman',
        'weight' : 'normal',
        'size'   : 14,
        }

X = np.array([0,1,3,4,8,14,17,24,27,30,32,33,34,35,37,41,42,47,50,65,70,74])
Y = np.array([2.101,2.298,2.320,2.331,2.352,2.360,2.373,2.444,2.455,2.554,2.563,2.585,2.600,2.603,2.712,2.745,2.763,2.784,2.831,2.833,2.834,2.843])

X1 = [-1,90]
Y1 = [2.233,2.233]
X2 = [-1,90]
Y2 = [2.375,2.375]


plt.figure(1)
ax1=plt.subplot(111)
ax1.plot(X,Y,color=(73/255,121/255,207/255),linewidth=2, ms=10, marker='x', label='Max Score in Training')
ax1.plot(X1,Y1,color="#c0504d",linewidth=2,linestyle='--', label="Max Random Score")
ax1.plot(X2,Y2,color=(105/255,203/255,100/255),linewidth=2,linestyle='--', label="Max Naive Strategy Score")
#ax1.plot(X3,Y3,color=(105/255,203/255,100/255),linewidth=2,linestyle='--', label="Average Reward from Out Best Strategy")
plt.grid(linestyle='dotted')
plt.xlim(-5,100)
plt.ylim(2,3)

ax1.legend(loc='center right')
ax1.set_xlabel('Epoches',fontdict=font)
ax1.set_ylabel('Score',fontdict=font)
savefig('score.eps',format="eps")
plt.show()