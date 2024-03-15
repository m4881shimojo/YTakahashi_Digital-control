        #digital control 
#デジタル制御　高橋安人
#20231205 shimojo
#20240228　見直し
#p104 Fig6-3
#ディジタルサーボのLQ制御
#　速度feedbackを入れてみる
# u(k)=-Gx
#n2,m2システムの応答を求める
#n2,m2システムのGは、p101v3_m2.pyで求める
#
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#from numpy.linalg import inv
#Digital Control 
n=2 #3次系
m=2 # m個のu入力
knum=30 #サンプル数
#
P=np.zeros((n,n));Q=np.zeros((n,m))
W0=np.zeros((n,n))
rinp=np.zeros(knum)

X=np.zeros((n,1))
X0=np.zeros((n,1))

u=np.zeros((knum,m))
U=np.zeros((m,1))
y=np.zeros((knum,2))
yv=np.array([[0],
             [0]])

# プラント
T=0.1 #sampling period
p=np.exp(-T)
#P,Q行列は、p57_p101.pyで計算する
P=np.array([[1.00000000, 0.09516258],
 [0.00000000, 0.90483742]])
Q=np.array([[0.10000000, 0.00483742],
 [0.00000000, 0.09516258]])

C=np.array([[1,0],
            [0,1]])
#
#---------------------
#################################################
#######     ここから応答の計算            ########
#################################################
#gain 計算すみのもの
#G=np.zeros((2,1))
#G=np.array([[K1,K2]])以下
#K1=22.324;K2=5.867;w=0.001  # case (1)
#K1=52.467;K2=9.470;w=0.0001 # case (2)
#K1=101.36784178;K2=13.51524203;0;w=0.00001# case (3)
#
#G=np.zeros((2,2))
############# gain G begin ####################
#  gain G
#G<--2x2 （p101v3_m2.py）で算出
G=np.array([[6.16910633, 0.4939516 ],
 [0.20049414, 5.53425101]]) ;w11=0.1

G=np.array([[6.16910633, 0.4939516 ],
 [0.20049414, 5.53425101]]) ;w11=0.01

G=np.array([[9.15862766, 0.49276541],
 [0.04466098, 8.63041919]]) ;w11=0.001

G=np.array([[9.90169817e+00, 4.91812390e-01],
 [5.23513142e-03, 9.40565610e+00]]) ;w11=0.0001
############# gain G end ####################

#使うGを以下に持ってくる
G=np.array([[6.16910633, 0.4939516 ],
 [0.20049414, 5.53425101]]) ;w11=0.01

#rinp(k)用
Xoff=np.array([[0.0],[0.0]])

ramp_start=20
for i in range(0,knum):
    if i<ramp_start: rinp[i]=1.0 #step input
    #elif i>=ramp_start: ramp[i]=1.0+(i-ramp_start)*0.016 #rump input
    elif i>=ramp_start: rinp[i]=1.0+T*(i-ramp_start) #rump input
#

for k in range(1,knum):    
    #input
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0    
    U=-np.dot(G,(X-Xoff)) # Uは(mx1)行列
    # ravel関数を適用させると、一次元のリストが返却
    u[k]=np.ravel(U)  #u[k,0]=U[0];u[k,1]=U[1]
    
    X=np.dot(P,X)+np.dot(Q,U)
    yv=np.dot(C,X)
    
    # ravel関数を適用させると、一次元のリストが返却
    y[k]=np.ravel(yv) #y[k,0]=yv[0];y[k,1]=yv[1]
    
 # 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
#plt.clf()
#plt.close()
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6,6)) # Figureの初期化
#1つの図に様々な大きさのグラフを追加
# https://pystyle.info/matplotlib-grid-sepc/
#縦方向に3つ場所を用意して、2つをss１に、1つをss2用に使う
#
gs = GridSpec(3, 1)  # 縦方向に3つ、横方向に１つの場所を用意
#ss1--> 場所は(0,0)、縦2つ、横１つ、を使用
ss1 = gs.new_subplotspec((0, 0), rowspan=2,colspan=1)  # ax1 を配置する領域
#ss2--> 場所は(2,0)、縦１つ横１つ、を使用
ss2 = gs.new_subplotspec((2, 0), rowspan=1, colspan=1)  # ax2 を配置する領域
#

t=np.arange(0,knum)
#########################################################
#                      ax1　PLOT                        #
#########################################################
ax1 = plt.subplot(ss1)
ax1.plot(t,y[:,0],'-*r',label="y(k)") 
ax1.plot(t,rinp,'--b',label="r(k)")  #input 

strg0="重み w={:.5g}".format(w11)
plt.title("図6-3 ディジタルサーボのLQ制御(m=2): "+strg0, fontname="MS Gothic")

Ymax=np.amax(y); Ymin=0.0
xp=knum*1/10; yp=Ymax*9/10  #plt.textの位置座標
strg1=" Gain:  K11={:.5g},  K12={:.5g}".format(G[0,0],G[0,1])
plt.text(xp,yp, strg1 ) #
xp=knum*1/10; yp=Ymax*8/10  #plt.textの位置座標
strg1=" Gain:  K21={:.5g},  K22={:.5g}".format(G[1,0],G[1,1])
plt.text(xp,yp, strg1 ) #

#plt.xlim(0,knum)
#plt.ylim(0,2)
plt.ylabel("Response ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い
#ax1.grid()

#########################################################
#                      ax2　PLOT                        #
#########################################################
ax2 = plt.subplot(ss2)
#ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,u[:,0],drawstyle='steps-post',color='g', linestyle='dashed', marker='o',label="u1(k)")
ax2.plot(t,u[:,1]*5,drawstyle='steps-post',color='b', linestyle='dashed', marker='o',label="u2(k)")

plt.ylabel("input")
plt.xlabel("step (k)")
plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い
#ax1.grid()

plt.tight_layout()
# 表示
plt.show()    