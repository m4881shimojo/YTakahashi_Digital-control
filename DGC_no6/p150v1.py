#digital control 
#デジタル制御　高橋安人
#20240408shimojo

#150 Fig8-1(a)
#同定及び適応制御
#8.1　簡単なサーボ系
#
#Eq.5-1 R(z):input Y(z):output V(z)=0
#G[z]=Gp*Gc Eq.5-16,Eq.5-17
#Y[k]=G[z]/(1+G[z])R[k]
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

#Digital Control 
n=3 #3次系
knum=60 #サンプル数
#　操作量の上限をきめる
ulimit=14/3. # |u|<ulimit
#ulimit=10. # |u|<ulimit
#
rinp=np.zeros(knum);y=np.zeros(knum)
e=np.zeros(knum);u=np.zeros(knum)
YY=np.zeros(knum)
m=2
p=np.zeros(m);q=np.zeros(m)
Kp=np.zeros(m);Kc=np.zeros(m);Kgain=np.zeros(m)
a=np.zeros(m);b=np.zeros(m)
a1=np.zeros(m);a2=np.zeros(m);a3=np.zeros(m)
b1=np.zeros(m);b2=np.zeros(m);b3=np.zeros(m)

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


#mple#input parameter
#a0=1.0/2.0 #負荷
#p149　根0.5±0.3j, 0
# Zi-----> 極
# F(z)=(z-z1)(z-z2)(z-z3)
# F(z)=z^3-(z1+z2+z3)z^2+(z1z2+z2z3+z3z1)z-z1z2z3
#F(z)=z^3-1.0z^2+0.34z
#F(z)=z^3+a_1*z^2+a_2*z^1+a_3
#then a_1=-1.0;a_2=0.34,;a_3=0
#Eq.(8-3)

#
#
F1=1.0-1.0+0.34  #F(1)
aa1=-1.0;aa2=0.34;aa3=0.0 
Tsample=0.1
for i in range(0,m):
    if i==0:a0=1.0
    if i==1:a0=1.0/2.0
    p[i]=np.exp(-a0*Tsample)
    q[i]=(1-p[i]-p[i]*a0*Tsample)/(p[i]+a0*Tsample-1)
    Kp[i]=(p[i]+a0*Tsample-1)/a0

for i in range(0,m):
    a[i]=(1/(p[i]+q[i]))*((aa1*q[i]+aa3)+q[i]*(1+p[i]-F1/(1+q[i]))) #Eq,(a),(b) P86
    Kgain[i]=aa1+1+p[i]-a[i] #Eq,(c) P87
    b[i]=F1/Kgain[i]/(1+q[i])-1 #Eq,(a),(b) (c)
    Kc[i]=Kgain[i]/Kp[i]

#---------------------
#G(z) AG,BG  閉ループゲイン
for i in range(0,m):
    a1[i]=a[i]-p[i]-1+Kgain[i]
    a2[i]=p[i]-a[i]*p[i]-a[i]+Kgain[i]*(b[i]+q[i])
    a3[i]=a[i]*p[i]+Kgain[i]*b[i]*q[i]
    b1[i]=Kgain[i]
    b2[i]=Kgain[i]*(q[i]+b[i])
    b3[i]=Kgain[i]*q[i]*b[i]
print("----------------無負荷----------------")
print("p={:.5g},Kp={:.5g},q={:.3g},Kgain={:.3g}".format(p[0],Kp[0],q[0],Kgain[0]))
print("Kc={:.5g},b={:.5g},a={:.3g}".format(Kc[0],b[0],a[0]))
print("----------------負荷----------------")
print("p={:.5g},Kp={:.5g},q={:.3g},Kgain={:.3g}".format(p[1],Kp[1],q[1],Kgain[1]))
print("Kc={:.5g},b={:.5g},a={:.3g}".format(Kc[1],b[1],a[1]))
#
#y(k)=-{a1y(k-1)+a2y(k-2)+a3y(k-3)}+(b1u(k-1)+b2u(k-2)+b3u(k-3))
#---------------------
#################################################
#######     ここから応答の計算            ########
#################################################
#
#input signal
#fig8-1用の入力
ylim=1.0;cyc=0.0;cycN=30
st1=5;st2=15;st3=20
for i in range(0,knum):
    cyc=np.floor(i/30.0)
    if i<=st1+cycN*cyc: rinp[i]=ylim/st1*(i-cycN*cyc)
    elif i<=st2+cycN*cyc: rinp[i]=ylim
    elif i<=st3+cycN*cyc: rinp[i]=ylim-ylim/st1*(i-st2-cycN*cyc)

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
cyc=0.0;cycN=30
st1=5;st2=15;st3=20
cyc=np.floor(i/30.0)

for k in range(0,knum):
    cyc=np.floor(k/30.0)
    #Phase 0;
    #初めにy[k]を計算  u[k-1]-->y[k] 2
    #k時点での状態変量の計算?
    #i=0 --> m=1　i=1 -->　m=2
    if k>=st2+cycN*cyc and  k < cycN+cycN*cyc:  
        i=1;alfa=1.
    else: i=0;alfa=1.

    #print(k,i,alfa)
    
    if k==0: y[k]=0.0
    elif k==1: y[k]=(1+p[i])*y[k-1]+Kp[i]*u[k-1]
    else: y[k]=(1+p[i])*y[k-1]-p[i]*y[k-2]+Kp[i]*u[k-1]+Kp[i]*q[i]*u[k-2]
    #YY[k]=y[k]
    #
    #    
    #Phase 1; e(k)=r(k)-y(k) 
    if k==0: e[k]=rinp[k]
    else: e[k]=rinp[k]-y[k]
       
    #Phase 2 Calcluate U(z)=Gc(z)E(z)
    #人刻み前の状態から定めた新たな入力u[k]を定める
    if k==0: u0=Kc[i]*e[k]*alfa
    else: u0=-a[i]*u[k-1]+Kc[i]*e[k]*alfa+Kc[i]*b[i]*e[k-1]*alfa
    if u0>ulimit: u0=ulimit
    elif u0<-ulimit: u0=-ulimit
    u[k]=u0
    
    #Phase 3 Calcluate Y(z)=Gp(z)U(z)   
    #u[k-1]---> u[k]に伴う、新たな状態量の計算
    # 私のイメージ的には u[k]-->y[k+1]と考えても良いか？
    #y[k]=(1+p)*y[k-1]-p*y[k-2]+Kp*u[k-1]+Kp*q*u[k-2]

    if k==0: y[k]=0.0
    elif k==1: y[k]=(1+p[i])*y[k-1]+Kp[i]*u[k-1]
    else: y[k]=(1+p[i])*y[k-1]-p[i]*y[k-2]+Kp[i]*u[k-1]+Kp[i]*q[i]*u[k-2]    
 
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
ax1.plot(t,rinp,'--b',label="r(k)") 

#Fig への値記入 
#strg0="a(1)={:.3g},a(2)={:.3g},a(3)={:.3g}".format(aa1,aa2,aa3)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(Tsample,Kp,p,q)
strg0="";strg1="";strg2=""
plt.title("図5-3 極を指定したdigital-servo系の挙動:"+strg0, fontname="MS Gothic")
#
Ymax=2.0; Ymin=0.0
xp=knum*1/10; yp=Ymax*9/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
plt.text(xp,yp-Ymax*1/10, strg2 ) 
plt.ylabel("response ")
plt.xlabel("step (k)")

plt.legend(loc='upper left')
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