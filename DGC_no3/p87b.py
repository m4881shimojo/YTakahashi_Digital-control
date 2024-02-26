#digital control 
#デジタル制御　高橋安人
#20231117shimojo (20240222見直し）

#p87(a) Fig5-3(b)
#Specified Poles: z1=0.5;z2=0.5+0.5j;z3=0.5-0.5j
#F(z)=z^3-1.5z^2+1.0z-0.25

#単一ループデジタル制御　追従系の設計
#Eq.5-1 R(z):input Y(z):output V(z)=0
#G[z]=Gp*Gc Eq.5-16,Eq.5-17
#Y[k]=G[z]/(1+G[z])R[k]
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

#Digital Control 
n=3 #3次系
knum=30 #サンプル数
#　操作量の上限をきめる
ulimit=14/3. # |u|<ulimit

# プラント
#Gp(z)=Kp(z+p)/((z-1)(z-p))    Eq.5-16
#Kp=p+T-1, q=(1-p-pT)/(p+T-1)  Eq.5-16
#制御アルゴリズム
#Gc(z)=Kc(z+b)/(z+a), |a|<1    Eq.5-17
# 1+Gp(z)Gc(z)------>F(z)
#
#F(z)=z^3+a(1)z^2+a(2)z+a(3)   Eq.5-18
# a(1)=a-p-1+K, K=KpKc
# a(2)=-ap-a+p+K(b+q)
# a(3)=ap+Kbq
#
# K(1+b)=F(1)/(1+q)-------(a)
# a=1/(p+q) [{a(1)q+a(3)}+q(1+p)-K(1+b)q]---(b)
# (b)式へ(a)式を代入すると、aが決まる
# するとKが決まる
# K=a(1)+1+p-a

#mple#input parameter
Tsample=0.1; p=np.exp(-Tsample)
q=(1-p-p*Tsample)/(p+Tsample-1)
Kp=p+Tsample-1

# Zi-----> 極
# F(z)=(z-z1)(z-z2)(z-z3)
#F(z)=z^3-1.5z^2+1.0z-0.25
F1=1.0-1.5+1.0-0.25  #F(1)　
#
r=np.zeros(knum);y=np.zeros(knum)
e=np.zeros(knum);u=np.zeros(knum)
YY=np.zeros(knum)
#
#
#F(z)=z^3+a_1*z^2+a_2*z^1+a_3
#Set poles of F(z),z1=0.5,z2=0.5+0.5j,z3=0.5-0.5j
#then a_1=-1.5;a_2=1.0,;a_3=-0.25
aa1=-1.5;aa2=1.0;aa3=-0.25 

a=(1/(p+q))*((aa1*q+aa3)+q*(1+p-F1/(1+q))) #Eq,(a),(b) P86
Kgain=aa1+1+p-a #Eq,(c) P87
b=F1/Kgain/(1+q)-1 #Eq,(a),(b) (c)
Kc=Kgain/Kp
print("a={:.5g},b={:.5g},K={:.3g},Kc={:.3g}".format(a,b,Kgain,Kc))
#---------------------
#G(z) AG,BG  閉ループゲイン
a1=a-p-1+Kgain
a2=p-a*p-a+Kgain*(b+q)
a3=a*p+Kgain*b*q
b1=Kgain
b2=Kgain*(q+b)
b3=Kgain*q*b
#
#y(k)=-{a1y(k-1)+a2y(k-2)+a3y(k-3)}+(b1u(k-1)+b2u(k-2)+b3u(k-3))
#---------------------
#################################################
#######     ここから応答の計算            ########
#################################################
#
#input signal
ramp_start=20
for i in range(0,knum):
    if i<ramp_start: r[i]=1.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    elif i>=ramp_start: r[i]=1.0+Tsample*(i-ramp_start) #rump input
#
#
#-------------------------------------------------
#Calculate response #
#--------------------------------------------------------
# Phase 0; u[k-1]--> y[k] k時点での状態量からの計算
# Phase 1; e(k)=r(k)-y(k) 偏差計算
# Phase 2; u[k]　新たな操作量を計算
# Phase 3: u[k] --> y[k] 入力を1刻み進めた状態量の計算
#
#これが正解かは怪しい！　正しいやり方はあるのだろう
#今回設定値rがあらかじめ決められているので上記計算が可能か。
#またTの値が変化すると結果が変わるのは腑に落ちない。計算式の不備？？
#入力uのリミッター上限値の値で応答は変わるから。ulimit=14/3
#---------------------------------------------------------
for k in range(0,knum):

    #Phase 0;
    #初めにy[k]を計算  u[k-1]-->y[k] 
    #k時点での状態変量の計算?

    if k==0: y[k]=0.0
    elif k==1: y[k]=(1+p)*y[k-1]+Kp*u[k-1]
    else: y[k]=(1+p)*y[k-1]-p*y[k-2]+Kp*u[k-1]+Kp*q*u[k-2]
    #YY[k]=y[k]
    #
    #    
    #Phase 1; e(k)=r(k)-y(k) 
    if k==0: e[k]=r[k]
    else: e[k]=r[k]-y[k]
       
    #Phase 2 Calcluate U(z)=Gc(z)E(z)
    #人刻み前の状態から定めた新たな入力u[k]を定める
    if k==0: u0=Kc*e[k]
    else: u0=-a*u[k-1]+Kc*e[k]+Kc*b*e[k-1]
    if u0>ulimit: u0=ulimit
    elif u0<-ulimit: u0=-ulimit
    u[k]=u0
    
    #Phase 3 Calcluate Y(z)=Gp(z)U(z)   
    #u[k-1]---> u[k]に伴う、新たな状態量の計算
    # 私のイメージ的には u[k]-->y[k+1]と考えても良いか？
    #y[k]=(1+p)*y[k-1]-p*y[k-2]+Kp*u[k-1]+Kp*q*u[k-2]

    if k==0: y[k]=0.0
    elif k==1: y[k]=(1+p)*y[k-1]+Kp*u[k-1]
    else: y[k]=(1+p)*y[k-1]-p*y[k-2]+Kp*u[k-1]+Kp*q*u[k-2]    
 
# 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6.5, 6.5)) # Figureの初期化
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
ax1.plot(t,r,'--b',label="r(k)") 

#Fig への値記入 
strg0="a(1)={:.3g},a(2)={:.3g},a(3)={:.3g}".format(aa1,aa2,aa3)
strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(Tsample,Kp,p,q)
plt.title("図5-3 極を指定したdigital-servo系の挙動:"+strg0, fontname="MS Gothic")
#
Ymax=2.0; Ymin=0.0
xp=knum*1/10; yp=Ymax*9/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
plt.text(xp,yp-Ymax*1/10, strg2 ) 
plt.ylabel("response ")
plt.xlabel("step (k)")

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
plt.legend(loc='upper right')
plt.grid() #ax2.grid() でも良い

plt.tight_layout()
# 表示
plt.show()    