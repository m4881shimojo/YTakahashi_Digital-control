#digital control 
#デジタル制御　高橋安人
#20231205 shimojo
#20240228 見直し
#p104 Fig6-3
#ディジタルサーボのLQ制御
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=2 #3次系
m=1 # m個のu入力
knum=30 #サンプル数
#
P=np.zeros((n,n));Q=np.zeros((n,m))
W0=np.zeros((n,n))
rinp=np.zeros(knum)#ramp入力

X=np.zeros((n,1));X0=np.zeros((n,1))
y=np.zeros((knum,1))

u=np.zeros((knum,m))
U=np.zeros((m,1))

# プラント
Tsample=0.1 #sampling period
p=np.exp(-Tsample)
#
P=np.array([[1,1-p],
            [0,p]]) #Eq.6-16
Q=np.array([[p+Tsample-1],
            [1-p]])
C=np.array([[1,0]])
#
#---------------------
#################################################
#######     ここから応答の計算            ########
#################################################

############# gain G begin ####################
#  gain G
#G<--1x2 （"p101v1.py"）で算出
#
#K1=8.36375858;K2= 3.26823936;w= 0.01
K1=22.324;K2=5.867;w=0.001  # case (1)
#K1=52.467;K2=9.470;w=0.0001 # case (2)
#K1=101.36784178;K2=13.51524203;0;w=0.00001# case (3)
############# gain G end ####################

#
G=np.array([[K1,K2]])
Xoff=np.array([[0.0],[0.0]])
#
ramp_start=20
for i in range(0,knum):
    if i<ramp_start: rinp[i]=1.0 #step input
    else: rinp[i]=1.0+Tsample*(i-ramp_start)*1.0 #rump input
#
    
for k in range(0,knum):    
    # x1=y, x2=dy/dt
    # x0--> offset レギュレータは(x1-x0)-->0だから
    #input
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0 #
    #u(k)=-(K1*(x1(k)-x0)+K2*X2(k))-->-(K1*x1(k)+K2*X2(k))+K1*x0           
    U[0]=-np.dot(G,(X-Xoff)) # Uは(mx1)行列
    u[k]=U[0] #u(k)はPLOT用
    #
    X=np.dot(P,X)+np.dot(Q,U)
    y[k]=np.dot(C,X)
    #X=P.dot(X)+Q.dot(U)
    #YY[k]=np.transpose(C.dot(X)) 
 # 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################

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
#ax1 = fig.add_subplot(2, 1, 1)

ax1.plot(t,y,'-*r',label="y(k)") 
ax1.plot(t,rinp,'--m',label="r(k)")  #input 

strg0="重み w={:.3g}".format(w)
plt.title("図6-3 ディジタルサーボのLQ制御 :"+strg0, fontname="MS Gothic")

Ymax=np.amax(y); Ymin=0.0
xp=knum*2/10; yp=Ymax*9/10  #plt.textの位置座標
strg1=" Gain:  K1={:.5g},  K2={:.5g}".format(G[0,0],G[0,1])
plt.text(xp,yp, strg1 ) #
#
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
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='o',label="u(k)")
plt.ylabel("input")
plt.xlabel("step (k)")

# 補助目盛を表示
plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
# 表示
plt.show()    