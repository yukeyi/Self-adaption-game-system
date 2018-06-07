from pylab import *

font = {
        'family' : 'times new roman',
        'weight' : 'normal',
        'size'   : 14,
        }

X = np.array([0,1,2,4,7,11,19,20,32,62,122,138,176,235,290,305,372,424,494,547])
Y = np.array([0.440,0.454,0.462,0.477,0.485,0.486,0.494,0.496,0.497,0.506,0.523,0.525,0.537,0.556,0.563,0.564,0.571,0.577,0.583,0.585])

X1 = [-15,550]
Y1 = [0.4475,0.4475]
X2 = [-15,550]
Y2 = [0.4303,0.4303]
X3 = [-15,550]
Y3 = [0.6607,0.6607]

plt.figure(1)
ax1=plt.subplot(111)
ax1.plot(X,Y,color=(73/255,121/255,207/255),linewidth=2, ms=10, marker='x', label='Max Score in Training')
ax1.plot(X1,Y1,color="#c0504d",linewidth=2,linestyle='--', label="Max Random Score")
ax1.plot(X2,Y2,color=(105/255,203/255,100/255),linewidth=2,linestyle='--', label="Max Naive Strategy Score")
#ax1.plot(X3,Y3,color=(105/255,203/255,100/255),linewidth=2,linestyle='--', label="Average Reward from Out Best Strategy")
plt.grid(linestyle='dotted')
plt.xlim(-20,600)
plt.ylim(0.4,0.6)

ax1.legend(loc='center right')
ax1.set_xlabel('Epoches',fontdict=font)
ax1.set_ylabel('Score',fontdict=font)
savefig('score.eps',format="eps")
plt.show()