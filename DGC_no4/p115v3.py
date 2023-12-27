#digital control 
#デジタル制御　高橋安人
#20231216 shimojo
#P行列がランク落ちで、E行列の逆行列が不可---＞失敗
#時系列データから求めるのは、原理的にできない？
#このため観測器を含めない！
#p115 6.4節　観測器を含むLQI制御
#時系列からの伝達関数-->可制御標準形の状態方程式
#可制御標準形（同伴形3.3節）の利用を主として使うことにする
#図6-7
#
#積分要素を入れるため、6.3節の手法を使う　u(k)---＞d(k)
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
# #https://analytics-note.xyz/programming/numpy-printoptions/
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=7,  floatmode="fixed")#　=True 指数表記禁止
#
n=7
m=1 # m個のu入力
#
knum=30 #収束計算の回数上限

P=np.zeros((n,n));Q=np.zeros((n,m))
G=np.zeros((m,n));H=np.zeros((n,n))
Eh=np.zeros((n,n))
W0=np.zeros((n,n));w=np.zeros((m,m))

ramp=np.zeros(knum)
X=np.zeros((n,1)) #状態変数

#u=np.ones((knum,m))
u=np.zeros(knum) #入力
d=np.ones((knum,m)) #刻み入力
UD=np.zeros((m,1))
y=np.zeros((knum,1))

#YY=np.zeros((knum,2)) #plot用data array. 行方向にdata
#
#図4-4　T1=10;T2=6;L=3のプロセス応答
#T=4 #sampling period
#T1=10;T2=6;r=T2/T1
#page 70 表4－5 書籍データから（これらを用いて応答を求めた）
#n=6
p0=0.708
g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108
#
a1=-p0
b1=g1;b2=g2-p0*g1;b3=g3-p0*g2;b4=g4-p0*g3;b5=g5-p0*g4;b6=g6-p0*g5
#
P=np.array([[1,0,b6,b5,b4,b3,b2-a1*b1],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,0,-a1]]) #7x7
Q=np.array([[b1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1]]) #7x1
#
C=np.array([[1,0,0,0,0,0,0]]) #1x7

##############################
#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C.T,C) #Eq.6-12　対角行列となる
Dw0=np.sum(np.abs(W0))
#
w11=0.8 # 重み w=np.zeros((m,m))　対角行列となる
w=np.eye(m,m)
w=np.dot(w11,w) # 重み(mxm)
#
print("\n -------------P, Q matrix------------")
#print("w={:.5g}".format(w))
print("P matrix")
print(P)
print("Q matrix")
print(Q)
print("C matrix")
print(C)
# 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
k=0; e0=1.0   #e0-->収束条件(for_loopでは利用せず)
#
while e0>1.0E-8 and k<1000: #収束チェック
    #
    k=k+1
    #E-->収束検証用
    Eh=np.copy(H) #E=H=E
    #
    #P'H(k)P--->A(working memo)
    A1=np.dot(P.T,np.dot(H,P))
    #P'H(k)q-->B(working memo)
    B1=np.dot(P.T,np.dot(H,Q))
    #(w+q'H(k)q)-->W1(working memo)
    W1=w+np.dot(Q.T,np.dot(H,Q))
    #(W1)^(-1)q'H(k)P-->G
    invW1=np.linalg.inv(W1) #逆行列を求める
    G=np.dot(invW1,np.dot(Q.T,np.dot(H,P))) #Eq. 6-14
    #
    H=A1-np.dot(B1,G)+W0  #Eq.6-13
    Eh=H-Eh; e0=np.sum(np.abs(Eh))/Dw0 #H(k)の収束状況を確認のためe0を計算
print("収束Num=",k);print("Gain=\n",G)

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
# 
#　ステップ＋ランプ入力
ramp_start=50;ramp_end=55
#
for i in range(0,knum):
    if i<ramp_start: ramp[i]=1.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    elif i>=ramp_start and i<=ramp_end: ramp[i]=1.0+(i-ramp_start)*0.1 #rump input
    #上記0.1は任意。ランプのスロープを調整する
    else:ramp[i]=1.0
ramp[15]=1.2# noise   

#
y[0]=0.;d[0]=0
for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    d[k]=-np.dot(G,X)+G[0,0]*ramp[k] # K0=G[0,0]
    UD[0]=d[k] #計算のため置き換え
    if k==0:u[k]=d[k]# For Plot
    else:u[k]=u[k-1]+d[k]
    #
    X=np.dot(P,X)+np.dot(Q,UD)
    y[k]=np.dot(C,X)
    #X=P.dot(X)+Q.dot(U)
    #
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
#####11111111111111111111########
# ax1　PLOT
#####11111111111111111111########
ax1 = plt.subplot(ss1)
#ax1 = fig.add_subplot(2, 1, 1)
#ax1.plot(t,YY,'-og') 
ax1.plot(t,y,'-*r') 
#ax1.plot(t,ramp,'--b')  #input 
ax1.plot(t,ramp,drawstyle='steps-post',color='b', linestyle='dashed', marker='.')

strg0="重み w={:.3g}".format(w11)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図6-7 LQIプロセス制御の例 :"+strg0, fontname="MS Gothic")

#Ymax=np.amax(y); Ymin=0.0
#xp=knum*3/10; yp=Ymax*8/10  #plt.textの位置座標
#strg1=" Gain: K1={:.5g},K2={:.5g}".format(G[0,0],G[0,1])
strg1=" Gain: "
#plt.text(xp,yp, strg1 ) #
#
#
#plt.xlim(0,knum)
#plt.ylim(0,2)
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
#
####222222222222222222222########
# ax2　PLOT
####222222222222222222222########
ax2 = plt.subplot(ss2)
#ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='.')
ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*')
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