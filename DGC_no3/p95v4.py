#digital control 
#デジタル制御　高橋安人
#20231120 shimojo 
#(見直し20240223) 
# 大体、80行以降がプログラム

#p95 List 5-1 PID制御
#ここではList5-1の状態式を用いない
# x(k+1)=Px(k)+Qu(k), y(k)=Cx(k)

#　実験データから伝達関数を求めたものを利用する
#4.4 時系列からの状態空間形
#p72T4.pyを参考

#図5-7の例は、page70で用いたモデルを利用。
#g1,g2,g3,g4,g5,g6,p　を利用した。
#何故かというと、書籍図5-7がそうだから。

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
knum=50
n=6
y=np.zeros(knum) # 出力
u=np.zeros(knum) #入力
d=np.zeros(knum) #1刻み入力
e=np.zeros(knum) #rinp[k]-y[k]
rinp=np.zeros(knum) #reference入力

#制御対象：むだ時間を含むプラントモデルのパラメータ
#(4-12),Fig 4-4　p66､p70
T1=10;T2=6;T=4;L1=3;N=0 #表4-5参照
L=N*T+L1
#page 70 表4－5 書籍データから
#n=6
#p=0.708　<--- p1_hat
#g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108
#
#acalculate gi.　p70&p35forT2Final.py （model値から求めたgi）
#g= [0.      0.00763 0.12794 0.1827  0.17224 0.14101 0.10764 0.07889 0.05634
# 0.03954 0.02742 0.01885 0.01287 0.00875 0.00593 0.00401 0.0027  0.00182
# 0.00122 0.00082 0.00055 0.00037 0.00025 0.00017 0.00011 0.00008 0.00005
# 0.00003 0.00002 0.00002 0.00001 0.00001 0.      0.      0.      0.

############################################
#　 4.4 時系列からの状態空間形　page 72 　  #
############################################
p1_hat=0.7078867 #書籍とほぼ同じ
#Dnum=6 #次数　書籍と同じ
g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108
 
P=np.array([[0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,0.,0.,0.,p1_hat]])
q=np.array([[g1], #p72に説明
            [g2],
            [g3],
            [g4],
            [g5],
            [g6]])
c=np.array([1.,0.,0.,0.,0.,0.]) #

#input signal
Noise_start=25 # #指令値を一回のみ変える。外乱とは言わない
#ramp_start=25
for i in range(0,knum):
    if i==Noise_start: rinp[i]=1#+0.5
    #elif i>=ramp_start: rinp[i]=1.0+(i-ramp_start)*0.016 #rump input
    else: rinp[i]=1.0
#

#PIDの各係数
Kp=1.5; Ki=0.5; Kd=0.5 #図5-7のパラメータ（書籍推薦）
#Kp=1.8; Ki=0.6; Kd=0.5  
#Kp=1.5; Ki=0.35; Kd=0.5 
#Kp=1.5; Ki=0.32; Kd=0.5 
#Kp=2.38; Ki=1.06; Kd=2.19 
#
#################################################
#######     ここから応答の計算            ########
#           X(k+1)=PX(k)+qu(K)                  #
#################################################
#-------------------------------------------------
#Calculate response # 応答の算定
Xk=np.zeros((n,1))  #x[k] clear
Xk1=np.zeros((n,1)) #x[k+1] clear
V=0 #noise

for k in range(0,knum):    
    #Phase 0;
    #初めにy[k]を計算
    y[k]=(np.dot(c,Xk)).item() #y[k]=cX(k)
    #warningが出るので”.item()”で要素の値を取得にした
    
    #Phase 1 e(k)=r(k)-y(k) 
    if k==0: e[k]=rinp[k]
    else: e[k]=rinp[k]-y[k]
    
    #Phase 2; d(k)=u(k)-u(k-1) 
    #1刻み前の状態から定めた新たな差分入力d[k]を定める
    #d[k]=Kp*(y[k-1]-y[k])+Ki*(r[k]-y[k])+Kd*(2.0*y[k-1]-y[k-2]-y[k])

    if k==0: d[k]=Kp*(-y[k])+Ki*(rinp[k]-y[k])+Kd*(-y[k])
    elif k==1: d[k]=Kp*(y[k-1]-y[k])+Ki*(rinp[k]-y[k])+Kd*(2.0*y[k-1]-y[k]) 
    else:      d[k]=Kp*(y[k-1]-y[k])+Ki*(rinp[k]-y[k])+Kd*(2.0*y[k-1]-y[k-2]-y[k])
    # 
    #入力u[k]を算出：u[k]=u[k-1]+d[k]
    if k==0: u[k]=d[k]
    else: u[k]=u[k-1]+d[k]

    #外乱入力はここに入れる？
    if k>=Noise_start:V=0.5 #step状の外乱とする（書籍）
    else:V=0
                    
    #Phase 3 Calcluate Y(z)=Gp(z)U(z)   
    #新たな状態量の計算   
    Xk1=np.dot(P,Xk)+np.dot(q,(u[k]+V))  #X(k+1)=PX(k)+qu(K)
    Xk=np.copy(Xk1)           #X(k)=X(k)
# 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(7, 7)) # Figureの初期化
#縦方向に3つ場所を用意して、2つ枠をss1、1つ枠をss2用に使う
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

ax1.plot(t,y,'-or',label="y(k)") 
ax1.plot(t,rinp,drawstyle='steps-post',color='m', linestyle='dashed', marker='',label="r(k)")  #input

#Fig への値記入
#strg0="Specified poles  a(1)={:.3g},a(2)={:.3g},a(3)={:.3g}".format(aa1,aa2,aa3)
strg0="Kp={:.3g},Ki={:.3g},Kd={:.3g}".format(Kp,Ki,Kd)
strg1="g1={:.3g},g2={:.3g},g3={:.3g},g4={:.3g},g5={:.3g},g6={:.3g},p={:.3g}".format(g1,g2,g3,g4,g5,g6,p1_hat)
strg2="実験データから伝達関数,T={:.3g}, T1={:.3g}, T2={:.3g}, L={:.3g}, L1={:.3g}".format(T,T1,T2,L,L1)
plt.title("図5-7 PID制御の例 :"+strg0, fontname="MS Gothic")
#
Ymax=1.6; Ymin=-0.1
#Ymax=2.0; Ymin=0
xp=knum*1/10; yp=Ymax*9/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
plt.text(xp,yp-Ymax*1/10, strg2, fontname="MS Gothic" ) 
#
plt.ylim(Ymin,Ymax)
plt.ylabel("Response ")
plt.xlabel("step (k)")

# 補助目盛を表示
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い
#ax1.grid()
#########################################################
#                      ax2　PLOT                        #
#########################################################
ax2 = plt.subplot(ss2)
#ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='+',label="u(k)")
ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="d(k)")
plt.ylabel("input")
plt.xlabel("step (k)")

# 補助目盛を表示
plt.minorticks_on()
#plt.legend(loc='center right')
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
# 表示
plt.show() 