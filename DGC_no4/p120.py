#digital control 
#デジタル制御　高橋安人
#
#6.5節 0型プラントのLQI制御
#p120　図6-9　20231222
#m=2入力系について
#
#制御対象--->　p57ver2 3連振動系
#p239　｢例７｣図A-3(d)の3連振動系を例として利用する
#　入力はステップ入力とした

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
knum=80 #計算総数
n=6 #次数
m=2 #入力数（今回は1入力のみとしている）

#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n))
#P=np.zeros((n,n));Q=np.zeros((n,m));Q0=np.eye(n,n)
P=np.eye(n,n);Q0=np.eye(n,n);Q=np.zeros((n,m))
Dpq=np.eye(n,n)
# A,B-->P,Qの中で使うからnp.eye(n,n)絶対

P1=np.zeros((n+m,n+m));Q1=np.zeros((n+m,m))
H=np.zeros((n+m,n+m));Eh=np.zeros((n+m,n+m))
W0=np.zeros((n+m,n+m));w=np.zeros((m,m))
G=np.zeros((m,n+m))
C1=np.zeros((m,n+1))
#応答の計算
ramp=np.zeros((knum,m))
Rinp=np.array([[0],[0]])
X=np.zeros((n+m,1)) #状態変数
u=np.zeros((knum,m)) #入力
d=np.ones((knum,m)) #刻み入力
U=np.zeros((m,1))
UD=np.zeros((m,1))
y=np.zeros((knum,m))

#観測器
#Eh--->H(k)の収束状況を確認
CP=np.zeros((m,n));CQ=np.zeros((m,m)) #temp
#Eobs=np.zeros((m*(n+m),n+m))
###

###

##############????????##########################

#観測器用
#YY=np.zeros((knum,3)) #PLOT用array 
#XX0=np.zeros((knum,n)) #PLOT用array
#XX=np.zeros((knum,n)) #PLOT用array

# #https://analytics-note.xyz/programming/numpy-printoptions/
np.set_printoptions(precision=4, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")

T=0.5 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#三連振動系
#4.1　プラント微分方程式と等価の差分式を参照のこと
A=np.array([[0,1.0,0.,0.,0.,0.],
            [-k1,-b1,k1,b1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [k1,b1,-(k1+k2),-(b1+b2),k2,b2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,k2,b2,-(k2+k3),-(b2+b3)]])
B=np.array([[0.,0.], #p58 hyou4-1
            [1.,0.],
            [0.,0.],
            [0.,0.],
            [0.,0.],
            [0.,2.]])
C=np.array([[1.,0.,0.,0.,0.,0.],
            [0.,0.,1.2,0.,0.,0.]])

print("-------------A, B and C matrix------------")
print("\nA matrix\n",A)
print("\nB matrix\n",Q)
print("\nC matrix\n",C)
#end
########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                 #
#    P=np.eye(n,n);Q0=np.eye(n,n);Dpq=np.eye(n,n) #<--MUST             #
######################################################################## 
#calculate P,Q matrix
#P=I+(AT)+1/2!(AT^2)+....  Eq.4-4
#Q0=T{I+1/2!(AT^2)+....}   Eq.4-5
#上記の反復計算
#list4-1 Pとqの算定
k=0;e0=0.000001;e1=1.0
A=T*A #T=0.5 #sampling time
while e1>e0:
    k=k+1
    Dpq=np.dot(A,Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))                                                                                                                                                                                                                                                                                                                                                                 
# get Q matrix
Q0=T*Q0; 
Q=np.dot(Q0,B) #Eq.4-4            
#calculate end

print("\n -------------P, Q matrix------------")
print("\nP matrix\n",P)
print("\nQ matrix\n",Q)
#
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
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.zeros((n+m,n+m))#Eq.6-12　対角行列となる
W0[0,0]=1.0;W0[1,1]=1.0 #y(k)の２つ要素とする
Dw0=np.sum(np.abs(W0)) #(m+n)x(m+n)
#
w=np.eye(m,m)
w11=0.1 # 重み(mxm)　m=2のため
w=np.dot(w11,w) # 重み(mxm)　m=2のため

## 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
k=0; e0=1.0   #e0-->収束条件
#
while e0>1.0E-8 and k<1000: #収束チェックの時使う
    #
    k=k+1
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

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2...] 計算済み    ######
#################################################
# 
#　ステップ＋ランプ入力
ramp_start=45#;ramp_end=55
#
for i in range(0,knum):
    ramp[i,0]=3.0;ramp[i,1]=1.0
    if i>ramp_start: ramp[i,0]=4.0;ramp[i,1]=2.0 

    #if i<ramp_start: ramp[i,0]=3.0;ramp[i,1]=1.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    #elif i>=ramp_start and i<=ramp_end:
    #    ramp[i,0]=3.0+T*(i-ramp_start)*0.5 #rump input
    #    ramp[i,1]=1.0+T*(i-ramp_start)*0.5 #rump input
    #上記0.5は任意。ランプのスロープを調整する
    #else:ramp[i,0]=3.0;ramp[i,1]=1.0
ramp[15,0]=4.0;ramp[20,1]=2.0

#　以下は外乱？としての初期値変化
#状態Xの初期値を与える。今回はn=3。
#X0=np.array([[0],[0],[0]])
#X=X+X0
#y[0]=0.;d[0]=0
C1=np.array([[1.,0,0,0,0,0,0,0],
             [0,1.,0,0,0,0,0,0]]) #1x3　mx(n+m)

X=np.zeros((n+m,1))
Xoff=np.array([[0],[0],[0],[0],[0],[0],[0],[0]])

for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    #入力R(k)をオフセットとして設定
    Xoff[0,0]=ramp[k,0]; Xoff[1,0]=ramp[k,1]
    UD=-np.dot(G,(X-Xoff)) 
    d[k]=UD.T #計算のため置き換え
    if k==0:u[k]=d[k]# For Plot
    else:u[k]=u[k-1]+d[k]
    #
    X=np.dot(P1,X)+np.dot(Q1,UD)
    #XX0[k]=np.transpose(X) #for PLOT
    y[k]=np.transpose(np.dot(C1,X))
    #X=P.dot(X)+Q.dot(U)
    #

########################################################################
#            次からは観測器による状態推定                       #
######################################################################## 
#多入出力系のobserverについてはここでは言及しない
#





#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#plt.clf()
#plt.close()
#
############################################################
#                figure 1                                  #
############################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8, 8)) # Figureの初期化
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
#ax1.plot(t,XX[:,0],'-*g') 
ax1.plot(t,y[:,0],'-*r') 
ax1.plot(t,y[:,1],'-*b') 
#ax1.plot(t,ramp,'--b')  #input 
ax1.plot(t,ramp,drawstyle='steps-post',color='g', linestyle='dashed', marker='')

strg0="重み w={:.3g}".format(w11)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図6-9 3連振動系の2変数LQI制御 :"+strg0, fontname="MS Gothic")

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
ax2.plot(t,u[:,0],drawstyle='steps-post',color='r', linestyle='dashed', marker='.')
ax2.plot(t,u[:,1],drawstyle='steps-post',color='b', linestyle='dashed', marker='.')
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
############################################################

# 表示
plt.show()    


#