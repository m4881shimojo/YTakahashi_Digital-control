#digital control 
#デジタル制御　高橋安人
#20231205 shimojo
#p104 Fig6-3
#ディジタルサーボのLQ制御
#n2,m2システムの応答を求める
#n2,m2システムのGは、p101v3_m2.pyで求める
#
#
import numpy as np
import math
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
ramp=np.ones(knum)

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
#K1=22.324;K2=5.867;w=0.001  # case (1)
#K1=52.467;K2=9.470;w=0.0001 # case (2)
#K1=101.36784178;K2=13.51524203;0;w=0.00001# case (3)
#G=np.zeros((2,1))
#G=np.array([[K1,K2]])以下4種類
G=np.array([[6.16910633, 0.4939516 ],
 [0.20049414, 5.53425101]]) ;w11=0.1
G=np.array([[6.16910633, 0.4939516 ],
 [0.20049414, 5.53425101]]) ;w11=0.01
G=np.array([[9.15862766, 0.49276541],
 [0.04466098, 8.63041919]]) ;w11=0.001
G=np.array([[9.90169817e+00, 4.91812390e-01],
 [5.23513142e-03, 9.40565610e+00]]) ;w11=0.0001

G=np.array([[6.16910633, 0.4939516 ],
 [0.20049414, 5.53425101]]) ;w11=0.01

ramp_start=20
for i in range(0,knum):
    if i<ramp_start: ramp[i]=1.0 #step input
    #elif i>=ramp_start: ramp[i]=1.0+(i-ramp_start)*0.016 #rump input
    elif i>=ramp_start: ramp[i]=1.0+T*(i-ramp_start) #rump input
#

for k in range(1,knum):    
    # x1=y, x2=dy/dt
    # x0--> offset レギュレータは(x1-x0)-->0だから
    #u(k)=-(K1*(x1(k)-x0)+K2*X2(k))-->-(K1*x1(k)+K2*X2(k))+K1*x0
    U=-np.dot(G,X)
    U[0]=U[0]+ramp[k]*G[0,0] # Uは(mx1)行列
    #
    u[k,0]=U[0];u[k,1]=U[1]#u(k)はPLOT用
    #
    X=np.dot(P,X)+np.dot(Q,U)
    yv=np.dot(C,X)
    y[k,0]=yv[0];y[k,1]=yv[1]
    #X=P.dot(X)+Q.dot(U)
    
 
 # 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
#plt.clf()
#plt.close()
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8, 8)) # Figureの初期化
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
# ax1　PLOT
ax1 = plt.subplot(ss1)
#ax1 = fig.add_subplot(2, 1, 1
#ax1.plot(t,YY,'-og') 
ax1.plot(t,y[:,0],'-*r') 
#ax1.plot(t,YY,'--y')  #input
#ax1.plot(t,e,'--k')  #input
ax1.plot(t,ramp,'--b')  #input 

#strg0="重み w={:.3g}".format(w)
strg0="重み w={:.5g}".format(w11)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図6-3 ディジタルサーボのLQ制御（速度FBあり） :"+strg0, fontname="MS Gothic")

Ymax=np.amax(y); Ymin=0.0
xp=knum*3/10; yp=Ymax*8/10  #plt.textの位置座標
strg1=" Gain: K1={:.5g},K2={:.5g}".format(G[0,0],G[0,1])
plt.text(xp,yp, strg1 ) #
#
#
#plt.xlim(0,knum)
plt.ylim(0,2)
plt.ylabel("Responce ")
plt.xlabel("step (k)")
#ax1.set_xticks(np.linspace(0, knum, 11))
#ax1.set_yticks(np.linspace(Ymin, Ymax,11))
# x軸に補助目盛線を設定
ax1.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# y軸に目盛線を設定
ax1.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# 補助目盛を表示
plt.minorticks_on()
plt.grid(which="minor", color="gray", linestyle=":")
#ax1.grid()

# ax2　PLOT
ax2 = plt.subplot(ss2)
#ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,u[:,0],drawstyle='steps-post',color='g', linestyle='dashed', marker='o')
ax2.plot(t,u[:,1]*5,drawstyle='steps-post',color='b', linestyle='dashed', marker='o')
plt.ylabel("Responce ")
plt.xlabel("step (k)")
ax2.set_xticks(np.linspace(0, knum, 11))
#ax2.set_yticks(np.linspace(-5,5,11))
# x軸に補助目盛線を設定
ax2.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# y軸に目盛線を設定
ax2.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# 補助目盛を表示
plt.minorticks_on()
plt.grid(which="minor", color="gray", linestyle=":")

plt.tight_layout()
# 表示
plt.show()    