#digital control 
#デジタル制御　高橋安人
#20231120 shimojo
#p95 List 5-1 PID制御
#ここではList5-1の状態式を用いない
# x(k+1)=Px(k)+Qu(k), y(k)=Cx(k)

#　理論式から伝達関数を求めたものを利用する
#図5-7の例は、page70で用いたモデルを利用。
#page70では伝達関数を導出した
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) 
#ここでは導出した伝達関数を使う

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
knum=50
y=np.zeros(knum)
u=np.zeros(knum)
d=np.zeros(knum)
e=np.zeros(knum)
rinp=np.zeros(knum)
g=np.zeros(knum) #gj=yj-y(j-1)

#from numpy.linalg import inv
#np.set_printoptions(formatter={'float': '{:.2f}'.format})
np.set_printoptions(precision=3, suppress=True)

#(4-12),Fig 4-4から選んだパラメータ
T1=10;T2=6
T=4;L1=3;N=0 #表4-5ではL=3-->N=0,L1=3
L=N*T+L1
#L=3
#K=1としてる
#G(z)=K(b1z^2+b2z+b3)/{z^(N+1)(z^2+s1z+a2)}
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常

print("\nN,T,L,L1 :",N,T,L,L1)
#係数を(4-12)を用いて決める
#
p1=np.exp(-T/T1);p2=np.exp(-T/T2)
a1=-(p1+p2);a2=p1*p2
d1=np.exp(L1/T1);d2=np.exp(L1/T2);r=T2/T1#r NOT EQ "0"
b1=1-(p1*d1-r*p2*d2)/(1-r)
b2=(p1*d1*(1+p2)-r*p2*d2*(1+p1))/(1-r)-(p1+p2)
b3=p1*p2*(1-(d1-r*d2)/(1-r))
#

#a1= -1.535262063651771 a2= 0.5866462195100318 # 
#a1=-1.535; a2=0.587#書籍

#b1= 0.00762904224602201 ;b2= 0.03840881647631877;b3=0.005346297135919452 # 
#b1=0.008;b2=0.038;b3=0.005 #書籍

#
print("\na1,a2 :",a1,a2) #;a_para=str(a1)+","+str(a2)
print("\nb1,b2,b3 :",b1,b2,b3) #;b_para=str(b1)+","+str(b2)+","+str(b3)

#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常
#z(^-1)が掛かっていると考えると
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-2]+b2*U[k-3]+b3*U[k-4]) 

#input signal
Noise_start=25 # ramp 入力はなし！
for i in range(0,knum):
    if i==Noise_start: rinp[i]=2.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    else: rinp[i]=1.0

#PIDの各係数
Kp=1.5; Ki=0.5; Kd=0.5 #図5-7のパラメータ
Kp=1.5; Ki=0.4; Kd=0.5 #図5-7のパラメータ
Kp=1.5; Ki=0.35; Kd=0.5 #図5-7のパラメータ
#Kp=2.38; Ki=1.06; Kd=2.19 #図5-7のパラメータ
#a1=-1.184;a2=0.344
#b1=0.008; b2=0.119; b3=0.034

#################################################
#######     ここから応答の計算            ########
#################################################
#-------------------------------------------------
#Calculate response #
#--------------------------------------------------------
# Phase 0; u[k-1]--> y[k] k時点での状態変数の計算
# Phase 3: u[k1] --> y[k] 入力を1刻み進めた状態変数の計算
#
#これが正解かは怪しい！　正しいやり方はあるのだろう
#---------------------------------------------------------

for k in range(0,knum):
    
    #Phase 0;
    #初めにy[k]を計算  u[k-1]-->y[k] 
    #以下は(4-14)式を基に求めた
    #   
    #y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2]+b2*u[k-3]+b3*u[k-4]) 
     
  

    if k==0:y[k]=0 
    elif k==1:y[k]=-(a1*y[k-1]) 
    elif k==2:y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2])  
    elif k==3:y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2]+b2*u[k-3])  
    else:     y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2]+b2*u[k-3]+b3*u[k-4]) 
    
    #YY[k]=Y[k]
    # 
    #Phase 1; e(k)=r(k)-y(k) 
    if k==0: e[k]=rinp[k]
    else: e[k]=rinp[k]-y[k]


    #Phase 2; d(k)=u(k)-u(k-1) 
    #人刻み前の状態から定めた新たな差分入力d[k]を定める
    #d[k]=Kp*(y[k-1]-y[k])+Ki*(r[k]-y[k])+Kd*(2.0*y[k-1]-y[k-2]-y[k])

    if k==0: d[k]=Kp*(-y[k])+Ki*(rinp[k]-y[k])+Kd*(-y[k])
    elif k==1: d[k]=Kp*(y[k-1]-y[k])+Ki*(rinp[k]-y[k])+Kd*(2.0*y[k-1]-y[k]) 
    else:      d[k]=Kp*(y[k-1]-y[k])+Ki*(rinp[k]-y[k])+Kd*(2.0*y[k-1]-y[k-2]-y[k])
    # 
    #入力u[k]とする
    #u[k]=u[k-1]+d[k]
    u[k]=u[k-1]+d[k]
                
    #Phase 3 Calcluate Y(z)=Gp(z)U(z)   
    #u[k-1]---> u[k]に伴う、新たな状態変数の計算
    # 私のイメージ的には u[k]-->y[k+1]と考えても良いか？
    #y[k]=p*y[k-1]+g1*u[k-1]+g2*u[k-2]+g3*u[k-3]+g4*u[k-4]+g5*u[k-5]+g6*u[k-6]-p*(g1*u[k-2]+g2*u[k-3]+g3*u[k-4]+g4*u[k-5]+g5*u[k-6])
   
    if k==0:y[k]=0 
    elif k==1:y[k]=-(a1*y[k-1]) 
    elif k==2:y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2])  
    elif k==3:y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2]+b2*u[k-3])  
    else: y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*u[k-2]+b2*u[k-3]+b3*u[k-4])
  
 

#########################################################
#　PLOT 
#########################################################
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
#ax1 = fig.add_subplot(2, 1, 1)
#ax1.plot(t,rinp,'-og') 
ax1.plot(t,y,'-or') 
ax1.plot(t,rinp,drawstyle='steps-post',color='c', linestyle='dashed', marker='.')  #input
#strg0="Specified poles  a(1)={:.3g},a(2)={:.3g},a(3)={:.3g}".format(aa1,aa2,aa3)
strg0="Kp={:.3g},Ki={:.3g},Kd={:.3g}".format(Kp,Ki,Kd)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(0,0,0,0)
strg2="理論式から伝達関数を求めたものを利用,T={:.3g}, T1={:.3g}, T2={:.3g}, L={:.3g}, L1={:.3g}".format(T,T1,T2,L,L1)
plt.title("図5-7 PID制御の例 :"+strg0, fontname="MS Gothic")
#
Ymax=2.0; Ymin=0.0
xp=knum*1/10; yp=Ymax*9/10  #plt.textの位置座標
plt.text(xp,yp, strg1, fontname="MS Gothic" ) #
plt.text(xp,yp-Ymax*1/10, strg2, fontname="MS Gothic" ) 
#plt.ylim(Ymin,Ymax,2.)
plt.ylabel("Responce ")
plt.xlabel("step (k)")
ax1.set_xticks(np.linspace(0, knum, 11))
ax1.set_yticks(np.linspace(Ymin, Ymax,11))
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
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='.')
ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='+')
plt.ylabel("Responce ")
plt.xlabel("step (k)")
ax2.set_xticks(np.linspace(0, knum, 11))
ax2.set_yticks(np.linspace(-1.0,2.0,11))
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