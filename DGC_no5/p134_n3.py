#digital control 
#デジタル制御　高橋安人
#
#6.5節 0型プラントのLQI制御
#p120　図6-9　20231222
#p134 Gp=1/z(^n) n=3

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
knum=40 #計算総数
n=3 #次数
m=1 #ここでは、入力数と積分動作+1が切り分けてないので注意
#
P=np.array([[0,1,0],
            [0,0,1],
            [0,0,0]])
Q=np.array([[0],
            [0],
            [1]])
C=np.array([[1,0,0]])

P1=np.zeros((n+m,n+m));Q1=np.zeros((n+m,m))
C1=np.zeros((m,n+m))
H=np.zeros((n+m,n+m));Eh=np.zeros((n+m,n+m))
W0=np.zeros((n+m,n+m));w=np.zeros((m,m))
G=np.zeros((m,n+m))
CP=np.zeros((m,n));CQ=np.zeros((m,m)) #temp

rinp=np.zeros(knum)
X=np.zeros((n+1,1)) #状態変数　積分動作で(n+1)
#
u=np.zeros((knum,m)) #入力
d=np.ones((knum,m)) #刻み入力
U=np.zeros((m,1))  #応答計算で利用
UD=np.zeros((m,1)) #応答計算で利用
VD=np.zeros((m,1)) #noise

y=np.zeros((knum,m)) #
E=np.zeros((n+1,n+1))
YY=np.zeros((knum,n+1)) #PLOT用array
XX0=np.zeros((knum,n+1)) #状態変数X　PLOT用array　
#
np.set_printoptions(precision=4, suppress=True)#　=True 指数表記禁止
#np.set_printoptions(precision=3,  floatmode="fixed")

print("-------------P, Q and C matrix------------")
print("\nP matrix\n",P)
print("\nQ matrix\n",Q)
print("\nC matrix\n",C)
#end
########################################################################
#            次からは P,Q 行列を P1,Q1 行列に変換する                    #
########################################################################
CP=np.dot(C,P);CQ=np.dot(C,Q) #Eq6-24
#####Generate P1 Matrix##########
# (11)領域  I単位行列
for i in range(0,m): 
    P1[i,i]=1.0 #P1はzerosの条件
#(12)領域
nn=0;mm=m #begin addr
for i in range(0,m):
    for j in range(0,n):
        P1[i+nn,j+mm]=CP[i,j]
#(21)領域
nn=m;mm=0 #begin addr
for i in range(0,n):
    for j in range(0,m):
        P1[i+nn,j+mm]=0
#(22)領域
nn=m;mm=m #begin addr
for i in range(0,n):
    for j in range(0,n):
        P1[i+nn,j+mm]=P[i,j]

#####Generate Q1 Matrix##########
#(1)領域
nn=0;mm=0 #begin addr
for i in range(0,m):
    for j in range(0,m):
        Q1[i+nn,j+mm]=CQ[i,j]

#(2)領域
nn=m;mm=0 #begin addr
for i in range(0,n):
    for j in range(0,m):
        Q1[i+nn,j+mm]=Q[i,j]

#P1,Q1 matrix
print("\n -------------P1 and Q1 matrix------------")
print("\nP1 matrix\n",P1)
print("\nQ1 matrix\n",Q1)
#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
#Begin Calculation
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数　　Eq.6-12
#　w   -->重み入力u

#W0の設定で結果が大きく変わるので注意
W0=np.zeros((n+m,n+m))#Eq.6-12　対角行列となる
W0[0,0]=1.0 #y(k)の１要素とする20240318
#W0[0,0]=1.0;W0[1,1]=1.0 #y(k)の２つ要素とする
#W0=np.eye(n+m,n+m) #対角行列となる
Dw0=np.sum(np.abs(W0)) #(m+n)x(m+n)
#
w11=0.1
w=np.eye(m,m)
w=np.dot(w11,w) # 重み(mxm)　対角
## 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
#k=0; e0=1.0   #e0-->収束条件
#
#while e0>1.0E-8 and k<1000: #収束チェックの時使う
for k in range(0,knum):
    #YY-->for PLOT
    YY[k,0]=G[0,0];YY[k,1]=G[0,1];YY[k,2]=G[0,2];YY[k,3]=G[0,3] 
    #
    #k=k+1 #while分の時使う
    #E-->収束検証用
    Eh=np.copy(H) #
    #
    #P'H(k)P--->A1(working memo)
    A1=np.dot(P1.T,np.dot(H,P1))
    #P'H(k)q-->B1(working memo)
    B1=np.dot(P1.T,np.dot(H,Q1))
    #(w+q'H(k)q)-->W1(working memo)
    W1=w+np.dot(Q1.T,np.dot(H,Q1))
    #(W1)^(-1)q'H(k)P-->G
    invW1=np.linalg.inv(W1) #逆行列を求める
    G=np.dot(invW1,np.dot(Q1.T,np.dot(H,P1))) #Eq. 6-14
    #
    H=A1-np.dot(B1,G)+W0  #Eq.6-13
    Eh=H-Eh; e0=np.sum(np.abs(Eh))/Dw0 #H(k)の収束状況を確認のためe0を計算
print("収束Num=",k);print("Gain=\n",G)
print(";w=",w11)

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
C1=np.array([[1,0,0,0]]) #P1からy(k)を取出す、
# 
#　ステップ＋ランプ入力
Noise_start=25 # #指令値を一回のみ変える。外乱とは言わない
ramp_start=500;ramp_end=550
#
for i in range(0,knum):
    if i<ramp_start: rinp[i]=1.0 #step input
    elif i>=ramp_start and i<=ramp_end: rinp[i]=1.0+(i-ramp_start)*0.1 #rump input
    #上記0.5は任意。ランプのスロープを調整する
    else:rinp[i]=1.0 

#　以下は外乱？としての初期値変化
#状態Xの初期値を与える。
X0=np.array([[0.0],[0.0],[0.0],[0.0]]) # 初期値
X=X+X0

#rinp(k)用 
Xoff=np.array([[0.0],[0.0],[0.0],[0.0]])
y[0]=0.;d[0]=0

for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0; Xoff[2,0]=0.0
    d[k]=-np.dot(G,(X-Xoff))
    #
    UD[0]=d[k] #計算のため置き換え
    if k==0:u[k]=d[k]# For Plot
    else:u[k]=u[k-1]+d[k]
    #
    #外乱入力はここに入れる？
    if k==Noise_start:VD[0]=0.5 #１刻みの差分入力のため
    else:VD[0]=0

    X=np.dot(P1,X)+np.dot(Q1,(UD+VD))
    XX0[k]=np.transpose(X) #for PLOT 状態変数X
    #
    y[k]=np.dot(C1,X)
    #X=P.dot(X)+Q.dot(U)
#########################################################
#　　　　　　　　　　　　　　　救根 　　　　　　　　　　　 #
#########################################################
#z^2-(2+1/w)+1
pol=np.poly1d([1,-(2+1/w11),1])
print("w=",w11,", 根= ",pol.r)

#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
############################################################
#                figure 1                                  #
############################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6.,6.)) # Figureの初期化
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
#ax1.plot(t,yopen,'-*c',label="yopen(k)") 
ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)")

strg0="重み w={:.3g}".format(w11)
plt.title("p.134 有限整定系の例（G(z)=1/z(^3)):"+strg0, fontname="MS Gothic")

#Ymax=np.amax(y); Ymin=0.0
strg1=" Gain: {:.5g},  {:.5g},  {:.5g},  {:.5g}".format(G[0,0],G[0,1],G[0,2],G[0,3])
#strg2=" FB  : {:.5g}, {:.5g}, {:.5g}".format(F[0,0],F[1,0],F[2,0])

Ymax=1.6; Ymin=-0.1
xp=knum*2/10; yp=Ymax*3/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
#plt.text(xp,yp-Ymax*1/10, strg2, fontname="MS Gothic" ) 

#plt.text(xp,yp, strg1 ) #
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
plt.ylabel("input")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
#
############################################################
#                figure 2                                  #
############################################################
fig = plt.figure(figsize=(6,5)) # Figureの初期化
ax2 = plt.subplot()
#
t=np.arange(0,knum)
ax2.plot(t,YY[:,0],'-*r',label="K1,w="+str(w))
ax2.plot(t,YY[:,1],'-*g',label="K2,w="+str(w))
#ax2.plot(t,YY[:,2],'-*b',label="K3,w="+str(w))

#strg0="収束反復計算回数={:.3g}, e0={:.3g}".format(Kok,e0)
strg0=""
plt.title("p.134 有限整定系（G(z)=1/z(^3)):Gain収束状況"+strg0, fontname="MS Gothic")
plt.ylabel("Gain K1, K2")
plt.xlabel("iterative calculation")
#
Ymax=np.amax(G); Ymin=0.0
Ymax=1.2; Ymin=-0.1
plt.ylim(Ymin,Ymax)
xp=knum*3/10; yp=Ymax*8.5/10  #plt.textの位置座標
strg1=" Gain: K1={:.5g},  K2={:.5g}".format(G[0,0],G[0,1])
plt.text(xp,yp, strg1 ) #

plt.minorticks_on()
plt.legend(loc='center right')
plt.grid() #ax1.grid() でも良い
#
plt.show()  