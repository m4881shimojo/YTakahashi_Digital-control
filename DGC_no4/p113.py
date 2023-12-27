#digital control 
#デジタル制御　高橋安人
#20231213 shimojo
#p111
#図6-5補償ブロックH(z)の応用
#
#図6-5はブロック線図の規約により図6-4から変形したもの。
#それでは両者は等価では？-->計算機上応答は同じになるはずでは
#書籍の表6-4では性能が低下したとするシミュレーション結果を出している
#
#私のこの計算結果では、同じ結果となった。一ステップずれているが..
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=3
m=1 # m個のu入力
knum=25 #計算の回数上限
P=np.zeros((n,n));Q=np.zeros((n,m))
G=np.zeros((m,n));H=np.zeros((n,n))
Eh=np.zeros((n,n))
W0=np.zeros((n,n)) #;w=np.zeros((m,m))

ramp=np.ones(knum)
X=np.zeros((n,1)) #状態変数

#u=np.ones((knum,m))

#d=np.ones((knum,m)) #刻み入力
#UD=np.zeros((m,1))
#y=np.zeros((knum,1))

x0=np.zeros(knum) #状態量
y=np.zeros(knum) #出力
e=np.zeros(knum) #入力
u=np.zeros(knum) #入力

#
YY=np.zeros((knum,2)) #plot用data array. 行方向にdata
############
#Eq.6-17
T=4 #sampling period
T1=10;T2=6;r=T2/T1
p1=np.exp(-T/T1);p2=np.exp(-T/T2) 
b1=1-(p1-r*p2)/(1-r)
b2=p1*p2-(p2-r*p1)/(1-r)
b3=p1*p2-(p2-r*p1)/(1-r)
p3=(b1*p2+b2)/(1-p2)
#
q1=-b1;q2=b1;q3=1-p2
#図5-3のシステム
#y[k]=x1[k],s1[k],s2[k]のシステム　Eq.6-21
P=np.array([[1,p1, p3],
            [0,p1,p3],
            [0,0,p2]])
Q=np.array([[q2],
            [q2],
            [q3]])
C=np.array([[1,0,0]])
#
#################################################
######　    ここからリカチ行列の収束計算     ######
######      およびGain G[K0,k1,k2]         ######
#################################################
#Begin Calculation
k=0; e0=1.0   #e0-->収束条件(for_loopでは利用せず)
Flg=True;Kok=-1 #e0-->収束条件のチェック
#
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C.T,C) #Eq.6-22　対角行列となる
Dw0=np.sum(np.abs(W0))
#
w=0.01 # 重み w=np.zeros((m,m))　対角行列となる
#w=0.1
#w=0.001
#
print("\n -------------P, Q matrix------------")
#print("w={:.5g}".format(w))
print("w=",w)
print("P matrix")
print(P)
print("Q matrix")
print(Q)
# 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
#
    #while e0>1.0E-8 or k<5:　#収束チェックの時使う
for k in range(0,knum):
    #print("k=",k,G)
    #YY-->for PLOT
    YY[k,0]=G[0,0]
    YY[k,1]=G[0,1] # G--> 2x1
    #
    #E-->収束検証用
    Eh=np.copy(H) #
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
    if (e0<1.0E-8) and Flg : Kok=k;Flg=False
    #収束条件チェック。但し、while_loopで使う条件。for_loopでなKnum回数計算する
    #End for k
    #
if Kok<0 :print("収束e0条件に達せずGain=\n",G)
else:print("収束Num=",Kok);print("Gain=",G)

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
# 
#　ステップ＋ランプ入力
ramp_start=20;ramp_end=50
#
for i in range(1,knum):
    if i<ramp_start: ramp[i]=1.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    elif i>=ramp_start and i<=ramp_end: ramp[i]=1.0+T*(i-ramp_start)*0.1 #rump input
    #上記0.1は任意。ランプのスロープを調整する
    else:ramp[i]=2.0
  
#Gain
K0=G[0,0];K1=G[0,1];K2=G[0,2]

y[0]=0.0;x0[0]=0
#積分で初期状態はゼロ？
x0[0]=K0*(ramp[0]-y[0])  #<--これではゼロでない
e[0]=x0[0]-K1*y[0]
u[0]=e[0]

for k in range(1,knum):
    #順序は大事
    #y[k]=(p1+p2)*y[k-1]-p1*p2*y[k-2]+b1*u[k-1]+b2*u[k-2]
    
    if k==1: y[k]=(p1+p2)*y[k-1]+b1*u[k-1]
    else:y[k]=(p1+p2)*y[k-1]-p1*p2*y[k-2]+b1*u[k-1]+b2*u[k-2]
    x0[k]=x0[k-1]+K0*(ramp[k]-y[k])
    e[k]=x0[k]-K1*y[k]
    u[k]=(p2-K2+p2*K2)*u[k-1]+e[k]-p2*e[k-1]  

    #
    print(k,y[k])
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
ax1.plot(t,ramp,'--b')  #input 
#ax1.plot(t,x0,'--g')  #input 
#ax1.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='o')

strg0="重み w={:.3g}".format(w)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図6-5(表) 補償ブロックH(z)の応用 :"+strg0, fontname="MS Gothic")

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
#ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*')
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