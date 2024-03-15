#digital control 
#デジタル制御　高橋安人
#20231216 shimojo
#20240305　見直し
#　方法２での状態方程式

#p115 6.4節　観測器を含むLQI制御
#時系列からの伝達関数からの状態方程式
#P72の方式を利用
#
#
#積分要素を入れるため、6.3節の手法を使う　u(k)---＞d(k)
#このままでは観測器observerは出来ない
#P行列がランク落ちで、E行列の逆行列が不可---＞失敗
#時系列データから求めるのは、原理的にできない？
#このため観測器を含めない！
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
# #https://analytics-note.xyz/programming/numpy-printoptions/
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=7,  floatmode="fixed")#　=True 指数表記禁止
#
n=7 #n=6だったのが積分器を加えたので+1した
m=1 # m個のu入力
#
knum=50 #収束計算の回数上限

P=np.zeros((n,n));Q=np.zeros((n,m))
G=np.zeros((m,n));H=np.zeros((n,n))
Eh=np.zeros((n,n))
W0=np.zeros((n,n));w=np.zeros((m,m))

rinp=np.zeros(knum)
X=np.zeros((n,1)) #状態変数

#u=np.ones((knum,m))
u=np.zeros((knum,m)) #入力
d=np.ones((knum,m)) #刻み入力
UD=np.zeros((m,1))
VD=np.zeros((m,1)) #noise
y=np.zeros((knum,1))

#YY=np.zeros((knum,2)) #plot用data array. 行方向にdata
#
#図4-4　T1=10;T2=6;L=3のプロセス応答
#T=4;L1=3 #sampling period
#T1=10;T2=6;r=T2/T1
#page 70 表4－5 書籍データから（これらを用いて応答を求めた）
#n=6
p0=0.708
g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108
#
a1=-p0
b1=g1;b2=g2-p0*g1;b3=g3-p0*g2;b4=g4-p0*g3;b5=g5-p0*g4;b6=g6-p0*g5
#
P=np.array([[1,0,1,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,0,-a1]]) #7x7
Q=np.array([[0],
            [g1],
            [g2],
            [g3],
            [g4],
            [g5],
            [g6]]) #7x1
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
w11=0.8# 重み w=np.zeros((m,m))　対角行列となる
w=np.eye(m,m)
w=np.dot(w11,w) # 重み(mxm)
#
print("\n -------------P, Q matrix------------")
print("w={:.5g}".format(w11))
print("P matrix\n",P)
print("Q matrix\n",Q)
# 
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
print(";w=",w11)

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
#書籍のgainだと、発信する
#G[0,0]=.845;G[0,1]=0.0;G[0,2]=.845;G[0,3]=.839
#G[0,4]=.745;G[0,5]=.572;G[0,6]=.551
# 
#　ステップ＋ランプ入力
Noise_start=25 # #指令値を一回のみ変える。外乱とは言わない
ramp_start=40;ramp_end=45
#
for i in range(0,knum):
    if i<ramp_start: rinp[i]=1.0 #step input
    #elif i>=ramp_start and i<=ramp_end: rinp[i]=1.0+(i-ramp_start)*0.1 #rump input
    #上記0.1は任意。ランプのスロープを調整する
    else:rinp[i]=1.0
  

#rinp(k)用 n=7
Xoff=np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
 
#
y[0]=0.;d[0]=0
for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0; Xoff[2,0]=0.0; Xoff[3,0]=0.0
    Xoff[4,0]=0.0; Xoff[5,0]=0.0; Xoff[6,0]=0.0

    d[k]=-np.dot(G,(X-Xoff))
    UD[0]=d[k] #計算のため置き換え

    if k==0:u[k]=d[k]# For Plot
    else:u[k]=u[k-1]+d[k]
    
    #外乱入力はここに入れる？
    if k==Noise_start:VD[0]=0.5 #１刻みの差分入力のため
    else:VD[0]=0
    #
    X=np.dot(P,X)+np.dot(Q,(UD+VD))
    y[k]=np.dot(C,X)
    #
#        
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
#plt.clf()
#plt.close()
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(7,7)) # Figureの初期化
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

ax1.plot(t,y,'-*r',label="y(k)") 
ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)")


#Fig への値記入
strg0="重み w={:.3g}".format(w11)
#strg0="実験データから伝達関数,T={:.3g}, T1={:.3g}, T2={:.3g}, L1={:.3g}".format(T,T1,T2,L1)
plt.title("図6-7 LQIプロセス制御の例(方法2):"+strg0, fontname="MS Gothic")

#Ymax=np.amax(y); Ymin=0.0
#xp=knum*3/10; yp=Ymax*8/10  #plt.textの位置座標
strg1="g1={:.3g},g2={:.3g},g3={:.3g},g4={:.3g},g5={:.3g},g6={:.3g},p={:.3g}".format(g1,g2,g3,g4,g5,g6,p0)
strg2=" Gain:{:.4g}, {:.4g}, {:.4g}, {:.4g}, {:.4g}, {:.4g}, {:.4g}".format(G[0,0],G[0,1],G[0,2],G[0,3],G[0,4],G[0,5],G[0,6])


Ymax=1.6; Ymin=-0.1
#Ymax=2.0; Ymin=0
xp=knum*0.5/10; yp=Ymax*9/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
plt.text(xp,yp-Ymax*1/10, strg2, fontname="MS Gothic" ) 


plt.ylim(Ymin,Ymax)
plt.ylabel("Response ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い
#
####222222222222222222222########
# ax2　PLOT
####222222222222222222222########
ax2 = plt.subplot(ss2)
#ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='.',label="u(k)")
ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="d(k)")
plt.ylabel("Responce ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
# 表示
plt.show()    