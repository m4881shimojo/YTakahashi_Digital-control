#digital control 
#デジタル制御　高橋安人
#20231120 shimojo
#p95 List 5-1 PID制御
#ここではList5-1の状態式を用いない
# x(k+1)=Px(k)+Qu(k), y(k)=Cx(k)

#　実験データから伝達関数を求めたものを利用する
#図5-7の例は、page70で用いたモデルを利用。
#page70では伝達関数を導出した
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) 
# 但し、プロセス特性は表4-5で求めた時系列形を用いる
#すなわち、g1,g2,g3,g4,g5,g6,p　を利用した。
#何故かというと、書籍図5-7がそうだから。


import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA

#
knum=50
y=np.zeros(knum);u=np.zeros(knum)
#Y=np.zeros(knum);U=np.zeros(knum)
d=np.zeros(knum)
e=np.zeros(knum)
rinp=np.zeros(knum)
g=np.zeros(knum) #gj=yj-y(j-1)
#np.set_printoptions(formatter={'float': '{:.2f}'.format})
np.set_printoptions(precision=3, suppress=True)

#(4-12),Fig 4-4から選んだパラメータ
#以下のパラメータは、本プログラムでは使用しない
T1=10;T2=6
T=4;L1=3;N=0 #表4-5ではL=3-->N=0,L1=3
L=N*T+L1

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
#
#以上不使用20231124

######################################################################
# ここから　プログラム本体
######################################################################
#input signal
Noise_start=25 # ramp 入力はなし！
for i in range(0,knum):
    if i==Noise_start: rinp[i]=2.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    else: rinp[i]=1.0
#

#PIDの各係数
Kp=1.5; Ki=0.5; Kd=0.5 #図5-7のパラメータ
#Kp=1.8; Ki=0.6; Kd=0.5  
#Kp=1.5; Ki=0.35; Kd=0.5 
#Kp=1.5; Ki=0.32; Kd=0.5 
#Kp=2.38; Ki=1.06; Kd=2.19 
#a1=-1.184;a2=0.344
#b1=0.008; b2=0.119; b3=0.034

#page 70 表4－5 書籍データから
n=6
p=0.708
g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108
#
#acalculate g()　p70&p35forT2Final.py
#g= [0.       0.       0.007629 0.127942 0.1827   0.172238 0.141007 0.107639
# 0.078888 0.056339 0.03954  0.027416 0.018846 0.012873 0.008752 0.00593
# 0.004008 0.002703 0.001821 0.001225 0.000823 0.000553 0.000371 0.000249
# 0.000167 0.000112 0.000075 0.00005  0.000034 0.000023]
#g1=0.007629;g2=0.127942;g3=0.1827;g4= 0.172238;g5= 0.141007;g6= 0.107639
#

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
    #y[k]=p*y[k-1]+g1*u[k-1]+g2*u[k-2]+g3*u[k-3]+g4*u[k-4]+g5*u[k-5]+g6*u[k-6]-p*(g1*u[k-2]+g2*u[k-3]+g3*u[k-4]+g4*u[k-5]+g5*u[k-6])
  
    if k==0:y[0]=0.0
    elif k==1:y[1]=p*y[0]+g1*u[0]
    elif k==2:y[2]=p*y[1]+g1*u[1]+g2*u[0]-p*(g1*u[0])
    elif k==3:y[3]=p*y[2]+g1*u[2]+g2*u[1]+g3*u[0]-p*(g1*u[1]+g2*u[0])
    elif k==4:y[4]=p*y[3]+g1*u[3]+g2*u[2]+g3*u[1]+g4*u[0]-p*(g1*u[2]+g2*u[1]+g3*u[0])
    elif k==5:y[5]=p*y[4]+g1*u[4]+g2*u[3]+g3*u[2]+g4*u[1]+g5*u[0]-p*(g1*u[3]+g2*u[2]+g3*u[1]+g4*u[0])
    else: y[k]=p*y[k-1]+g1*u[k-1]+g2*u[k-2]+g3*u[k-3]+g4*u[k-4]+g5*u[k-5]+g6*u[k-6]-p*(g1*u[k-2]+g2*u[k-3]+g3*u[k-4]+g4*u[k-5]+g5*u[k-6])
 
     # 
    #Phase 1 e(k)=r(k)-y(k) 
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
   
    if k==0:y[0]=0.0
    elif k==1:y[1]=p*y[0]+g1*u[0]
    elif k==2:y[2]=p*y[1]+g1*u[1]+g2*u[0]-p*(g1*u[0])
    elif k==3:y[3]=p*y[2]+g1*u[2]+g2*u[1]+g3*u[0]-p*(g1*u[1]+g2*u[0])
    elif k==4:y[4]=p*y[3]+g1*u[3]+g2*u[2]+g3*u[1]+g4*u[0]-p*(g1*u[2]+g2*u[1]+g3*u[0])
    elif k==5:y[5]=p*y[4]+g1*u[4]+g2*u[3]+g3*u[2]+g4*u[1]+g5*u[0]-p*(g1*u[3]+g2*u[2]+g3*u[1]+g4*u[0])
    else: y[k]=p*y[k-1]+g1*u[k-1]+g2*u[k-2]+g3*u[k-3]+g4*u[k-4]+g5*u[k-5]+g6*u[k-6]-p*(g1*u[k-2]+g2*u[k-3]+g3*u[k-4]+g4*u[k-5]+g5*u[k-6])

    # 
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
#ax1.plot(t,u,'--y')  #input
#ax1.plot(t,e,'--k')  #input
ax1.plot(t,rinp,drawstyle='steps-post',color='c', linestyle='dashed', marker='.')  #input

#strg0="Specified poles  a(1)={:.3g},a(2)={:.3g},a(3)={:.3g}".format(aa1,aa2,aa3)
strg0="Kp={:.3g},Ki={:.3g},Kd={:.3g}".format(Kp,Ki,Kd)
strg1="g1={:.3g},g2={:.3g},g3={:.3g},g4={:.3g},g5={:.3g},g6={:.3g},p={:.3g}".format(g1,g2,g3,g4,g5,g6,p)
strg2="実験データから伝達関数を求めたものを利用,T={:.3g}, T1={:.3g}, T2={:.3g}, L={:.3g}, L1={:.3g}".format(T,T1,T2,L,L1)
plt.title("図5-7 PID制御の例 :"+strg0, fontname="MS Gothic")
#
Ymax=2.0; Ymin=0.0
xp=knum*1/10; yp=Ymax*9/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
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
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='+')
ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*')
plt.ylabel("Responce ")
plt.xlabel("step (k)")
ax2.set_xticks(np.linspace(0, knum, 11))
ax2.set_yticks(np.linspace(-0.5,1.5,11))
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