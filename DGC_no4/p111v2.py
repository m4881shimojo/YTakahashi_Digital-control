#digital control 
#デジタル制御　高橋安人
#20231213 shimojo
#20240304　見直し
#p111
##表6-4 LQI制御例
#表6-3　成功例を用いた LQI制御
#
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=3
m=1 # m個のu入力
knum=40 #計算の回数上限

P=np.zeros((n,n));Q=np.zeros((n,m))
G=np.zeros((m,n));H=np.zeros((n,n))
Eh=np.zeros((n,n))
W0=np.zeros((n,n));w=np.zeros((m,m))

rinp=np.zeros(knum)
X=np.zeros((n,1)) #状態変数
X0=np.zeros((n,1))

#u=np.ones((knum,m))
u=np.zeros((knum,m)) #入力
d=np.ones((knum,m)) #刻み入力
UD=np.zeros((m,1))
y=np.zeros((knum,1))

#
YY=np.zeros((knum,n)) #plot用data array. 行方向にdata
############
#Eq.6-17
Tsample=4 #sampling period
T1=10;T2=6;r=T2/T1
p1=np.exp(-Tsample/T1);p2=np.exp(-Tsample/T2) 
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
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C.T,C) #Eq.6-22　対角行列となる
Dw0=np.sum(np.abs(W0))
#
#w11=0.1 # 重み w=np.zeros((m,m))　
w11=0.01 # 重み w=np.zeros((m,m))
w=np.eye(m,m)
w=np.dot(w11,w) # 重み(mxm)
#
print("\n -------------P, Q matrix------------")
print("w={:.5g}".format(w11))
print("P matrix\n",P)
print("Q matrix\n",Q)
# 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
k=0; e0=1.0   #e0-->収束条件(for_loopでは利用せず)
Flg=True;Kok=-1 #e0-->収束条件のチェック
#
#while e0>1.0E-8 or k<5:　#収束チェックの時使う
for k in range(0,knum):
    #YY-->for PLOT
    YY[k,0]=G[0,0]
    YY[k,1]=G[0,1] # G--> 3x1
    YY[k,2]=G[0,2] # G--> 3x1
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
print(";w=",w11)
# 

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
# 
#　ステップ＋ランプ入力
ramp_start=10;ramp_end=15
for i in range(0,knum):
    if i<ramp_start: rinp[i]=1.0 #step input
    elif i>=ramp_start and i<=ramp_end: rinp[i]=1.0+Tsample*(i-ramp_start)*0.016 #rump input
    #上記0.016は任意。ランプのスロープを調整する
    else:rinp[i]=1.5

#rinp(k)用
Xoff=np.array([[0.0],[0.0],[0.0]])
#
y[0]=0.;d[0]=0
for k in range(1,knum): 
    #d[k] 積分が入ってると、こうなるのかな？
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0; Xoff[2,0]=0.0 
    d[k]=-np.dot(G,(X-Xoff))

    UD[0]=d[k] #計算のため置き換え
    #u[k]=np.ravel(u[k-1]+d[k]) # For Plot
    u[k]=u[k-1]+d[k] # For Plot
    #
    X=np.dot(P,X)+np.dot(Q,UD)
    y[k]=np.dot(C,X)
    #
    print(k,y[k])
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
#####11111111111111111111########
# ax1　PLOT
#####11111111111111111111########
ax1 = plt.subplot(ss1)
ax1.plot(t,y,'-*r',label="y(k)") 
#ax1.plot(t,rinp,'--b',label="r(k)")  #input
ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)") 

strg0="重み w={:.3g}".format(w11)
plt.title("図6-4(表) プロセスのLQI制御 :"+strg0, fontname="MS Gothic")

Ymax=np.amax(y); Ymin=0.0
xp=knum*3/10; yp=Ymax*5/10  #plt.textの位置座標
strg1=" Gain: K0={:.5g}, K1={:.5g}, K2={:.5g}".format(G[0,0],G[0,1],G[0,2])
plt.text(xp,yp, strg1 ) #
#
#
#plt.xlim(0,knum)
#plt.ylim(0,2)
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
plt.ylabel("input ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
# 表示
plt.show()    