from pylab import *

font = {
        'family' : 'times new roman',
        'weight' : 'normal',
        'size'   : 14,
        }


plt.figure(1)
ax1=plt.subplot(111)

ans = 0
for i in range(14):
    ans += 1/(i+1)
Y1 = [1.123,0.997,0.953,0.896,0.780,0.751,0.639,0.596,0.592,0.598,0.667,0.757,0.692,0.683]
for iter in range(len(Y1)):
    Y1[iter] = Y1[iter]*ans

X = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
X1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

ax1.bar(X1, Y1, facecolor=(73/255,121/255,207/255), edgecolor='white',width=0.6,alpha=1, label="Best Model")

for iter in range(len(Y1)):
    plt.text(X1[iter],Y1[iter]+0.1,'%.2f' % Y1[iter],ha='center',va='top')

ax1.set_ylim((1.6,4))
ax1.set_xticks(X)
ax1.set_xlabel('Tokens Read',fontdict=font)
ax1.set_ylabel('Accuracy',fontdict=font)
ax1.legend(loc='upper right')
ax1.set_xlabel('Action Differences',fontdict=font)
ax1.set_ylabel('Average Reward',fontdict=font)
savefig('score2.eps',format="eps")
plt.show()