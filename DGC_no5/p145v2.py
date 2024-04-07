#digital control 
#デジタル制御　高橋安人
#7.5 多変数制御系　20240331
#p145図7-8
#入力　u1,u2を選ぶ(line138あたり)
#p59v2.pyより3連振動系
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA

n=6 #次数
m=2 #2入力2出力
knum=30 #サンプリング総数
#
Tsample=0.5 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n))
P=np.eye(n,n);Q0=np.eye(n,n);Dpq=np.eye(n,n)
# Gain Matrix
K=np.zeros((m,n));F=np.zeros((m,m))
In=np.eye(n,n);Im=np.eye(m,m)

#W=np.zeros(6);Z=np.zeros(6)
A=np.array([[0,1.0,0.,0.,0.,0.],
            [-k1,-b1,k1,b1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [k1,b1,-(k1+k2),-(b1+b2),k2,b2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,k2,b2,-(k2+k3),-(b2+b3)]])
B=np.array([[0.,0.],
            [1.,0.],
            [0.,0.],
            [0.,0.],# 0,1,0 <- change 20231028
            [0.,0.],
            [0.,2.]]) #<=2.0
C=np.array([[1.,0.,0.,0.,0.,0.],
            [0.,0.,1.2,0.,0.,0.]])
#
#応答の計算のためのarray
rinp=np.zeros((knum,m)) #r(k)

X=np.zeros((n,1)) #状態変数
u=np.zeros((knum,m)) #入力
#
UD=np.zeros((m,1))
y=np.zeros((knum,m))#PLOT

########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                  #
######################################################################## 
#calculate P,Q matrix
#list4-1 Pとqの算定
k=0;e0=1.0E-8;e1=1.0 #誤差許容値
A1=Tsample*A #A1---> working memo
while e1>e0:
    k=k+1
    Dpq=A1.dot(Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))
    #
# while end
# get Q matrix          
Q0=Tsample*Q0; Q=np.dot(Q0,B)            
#calculate end

# check calcutated matric P,Q
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
#print("number of recursion=",k)
print("\nP matrix\n",P)
print("\nQ matrix\n",Q)
#end

#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
#Begin Calculation
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C.T,C) #Eq.6-12　対角行列となる
Dw0=np.sum(np.abs(W0)) #(m+n)x(m+n)
#
w=np.eye(m,m)
w11=0.1 # 重み(mxm)　m=2のため
w11=0.01 # 重み(mxm)　m=2のため
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
    A1=np.dot(P.T,np.dot(H,P))
    #P'H(k)q-->B1(working memo)
    B1=np.dot(P.T,np.dot(H,Q))
    #(w+q'H(k)q)-->W1(working memo)
    W1=w+np.dot(Q.T,np.dot(H,Q))
    #(W1)^(-1)q'H(k)P-->G
    invW1=np.linalg.inv(W1) #逆行列を求める
    K=np.dot(invW1,np.dot(Q.T,np.dot(H,P))) #Eq. 6-14
    #
    H=A1-np.dot(B1,K)+W0  #Eq.6-13
    Eh=H-Eh; e0=np.sum(np.abs(Eh))/Dw0 #H(k)の収束状況を確認のためe0を計算
print("収束Num=",k);print("Gain=\n",K)
print(";w=",w11)

#-----------------------------------------------------------#
#dc gain
#
invIP=LA.inv((In-P))   
G1=np.dot(np.dot(C,invIP),Q)
print("dc gain=",G1)

# F gain
invIP=LA.inv((In-P))
F1=(Im+np.dot(K,np.dot(invIP,Q)))#F1 working
F=np.dot(F1,LA.inv(G1))
print("F gain=",F)

#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2...] 計算済み    ######
#################################################
# 
#　ステップ＋ランプ入力
Sup1=30;Sup2=40 #今回はStep_Upは行わない
Noise1=100;Noise2=120 #今回ノイズは範囲外とした
#
#今回はstep状変化とする
for i in range(0,knum):
    #rinp[i,0]=3.0;rinp[i,1]=1.0
    rinp[i,0]=1.0;rinp[i,1]=0.0 #u1 のみ
    #rinp[i,0]=0.0;rinp[i,1]=1.0 #u2 のみ

    #if i>Sup1: rinp[i,1]=2.0 #u2 step_UP
    #if i>Sup2: rinp[i,0]=4.0 #u1 step_UP
    
#状態Xの初期値を与える。
#X0=np.array([[0],...,[0],[0]]) #n+m
#X=X+X0
#y[0]=0.;d[0]=0
C1=np.array([[1.,0,0,0,0,0],
             [0,1.,0,0,0,0]]) #m*(n+m)

#X=np.zeros((n+m,1))
#入力R(k)をオフセットとして設定するため
#Xoff=np.array([[0],[0],[0],[0],[0],[0]])

win=np.zeros((m,1)) #ここで新たに入力として導入した変数(p144参照)

for k in range(1,knum): #
    #winに入力値としてy1,y2の値を入れる
    win[0,0]=rinp[k,0]; win[1,0]=rinp[k,1]
    UD=np.dot(F,win)-np.dot(K,X) #p144 (d)式
    u[k]=np.transpose(UD) # PLOT用に収納

    X=np.dot(P,X)+np.dot(Q,(UD))
    y[k]=np.transpose(np.dot(C,X))
    #
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
############################################################
#                figure 1                                  #
############################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6,6)) # Figureの初期化
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
ax1.plot(t,y[:,0],'-*r',label="y1_step(k)") 
ax1.plot(t,y[:,1],'-*m',label="y2_step(k)") 

ax1.plot(t,rinp[:,0],drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r1(k)")
ax1.plot(t,rinp[:,1],drawstyle='steps-post',color='c', linestyle='dashed', marker='',label="r2(k)")

strg0="重み w={:.3g}".format(w11)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図7-8 対角項優位にした振動体のステップ応答："+strg0, fontname="MS Gothic")

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
ax2.plot(t,u[:,0],drawstyle='steps-post',color='g', linestyle='dashed', marker='.',label="u1(k)")
ax2.plot(t,u[:,1],drawstyle='steps-post',color='b', linestyle='dashed', marker='.',label="u2(k)")

plt.ylabel("input")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
#
plt.show()  
