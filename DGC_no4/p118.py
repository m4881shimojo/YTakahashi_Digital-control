#digital control 
#デジタル制御　高橋安人
#20231222 shimojo
#p118 6.5節　０型プラントのLQI制御
#積分要素を入れるための手法の一般化　u(k)---＞d(k)
#
#図6-8
#
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
# #https://analytics-note.xyz/programming/numpy-printoptions/
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=7,  floatmode="fixed")#　=True 指数表記禁止
#
n=2
m=1 # m個のu入力
#
knum=40 #収束計算の回数上限

P=np.zeros((n,n));Q=np.zeros((n,m))
C=np.zeros((m,n))
P1=np.zeros((n+1,n+1));Q1=np.zeros((n+1,m))
C1=np.zeros((m,n+1))

Eh=np.zeros((n+1,n+1));H=np.zeros((n+1,n+1))
G=np.zeros((m,n+1))
W0=np.zeros((n+1,n+1)) #;w=np.zeros((m,m))

ramp=np.zeros(knum)
X=np.zeros((n+1,1)) #状態変数

#u=np.ones((knum,m))
u=np.zeros(knum) #入力
d=np.ones((knum,m)) #刻み入力
dob=np.ones((knum,m))#observar用

U=np.zeros((m,1))
UD=np.zeros((m,1))
y=np.zeros((knum,1))


#観測器用
y=np.zeros(knum)
E=np.zeros((n+1,n+1))
#YY=np.zeros(knum,n+1) #PLOT用array
XX0=np.zeros((knum,n+1)) #PLOT用array
XX=np.zeros((knum,n+1)) #PLOT用array
#
#p118 振動系
a1=-1.866; a2=0.933
b0=0.0; b1=0.067
#
P=np.array([[0,1],
            [-a2,-a1]]) #
Q=np.array([[0],
            [1]]) #2x1
#
C=np.array([[b1,b0]]) #1x2
#LQI制御
P1=np.array([[1,-a1*b0,b1-a1*b0],
             [0,0,1],
            [0,-a2,-a1]]) #3x3
Q1=np.array([[b0],
            [0],
            [1]]) #3x1
#
C1=np.array([[1,0,0]]) #1x3

##############################
#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C1.T,C1) #Eq.6-12　対角行列となる
Dw0=np.sum(np.abs(W0))
#
w11=0.1 # 重み w=np.zeros((m,m))　対角行列となる
w=np.eye(m,m)
w=np.dot(w11,w) # 重み(mxm)
#
#
print("\n -------------P, Q matrix------------")
#print("w={:.5g}".format(w))
print("w=",w)
print("P matrix")
print(P1)
print("Q matrix")
print(Q1)
print("C matrix")
print(C1)
# 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
k=0; e0=1.0   #e0-->収束条件(for_loopでは利用せず)
#
while e0>1.0E-8 and k<1000: #収束チェックの時使う
    #
    k=k+1
    #E-->収束検証用
    Eh=np.copy(H) #E=H=E
    #
    #P'H(k)P--->A(working memo)
    A1=np.dot(P1.T,np.dot(H,P1))
    #P'H(k)q-->B(working memo)
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
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
# 
#　ステップ＋ランプ入力
ramp_start=50;ramp_end=55
ramp_start=50000;ramp_end=55000
#
for i in range(0,knum):
    if i<ramp_start: ramp[i]=1.0 #step input
    #elif i>=ramp_start: r[i]=1.0+(i-ramp_start)*0.016 #rump input
    elif i>=ramp_start and i<=ramp_end: ramp[i]=1.0+(i-ramp_start)*0.5 #rump input
    #上記0.5は任意。ランプのスロープを調整する
    else:ramp[i]=1.0
ramp[20]=1.5# noise   

#　以下は外乱？としての初期値変化
#状態Xの初期値を与える。今回はn=3。
X0=np.array([[0],[0],[0]])
X=X+X0

y[0]=0.;d[0]=0
for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    d[k]=-np.dot(G,X)+G[0,0]*ramp[k] # K0=G[0,0]
    UD[0]=d[k] #計算のため置き換え
    if k==0:u[k]=d[k]# For Plot
    else:u[k]=u[k-1]+d[k]
    #
    X=np.dot(P1,X)+np.dot(Q1,UD)
    XX0[k]=np.transpose(X) #for PLOT
    y[k]=np.dot(C1,X)
    #X=P.dot(X)+Q.dot(U)
    #
#open loop
yopen=np.zeros(knum) 
uopen=np.zeros(knum) 
Xopen=np.zeros((n,1)) #状態ベクトルX=np.zeros((n,1)) #状態ベクトル
Uop=np.zeros((m,1))
#yopen[0]=0.;uopen[0]=0

for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    uopen[k]=ramp[k]
    Uop[0]=uopen[k] #計算のため置き換え
     #
    Xopen=np.dot(P,Xopen)+np.dot(Q,Uop)
    yopen[k]=np.dot(C,Xopen)
    #X=P.dot(X)+Q.dot(U)
#        
########################################################################
#            次からは観測器による状態推定                       #
######################################################################## 
x_hat=np.zeros((knum,n+1))
#x_hat0=np.zeros((knum,n))
XH=np.zeros((n+1,1)) #状態ベクトルx_hat
XH0=np.zeros((n+1,1)) #状態ベクトルx_hat0

#Calculate E（可観測行列 Eq.3-24）
#F(P)=P^n+α1P(n-1)＋．．．αnI
#F(P)=P^n　（有限整定観測器、αn=0）とした
#
f0=np.array([[0.],[0.],[1.]]) #3x1
E0=np.array((m,n+1)) 
P0=np.eye(n+1,n+1)      
for j in range(0,n+1):
    P0=np.dot(P0,P1)
    E0=np.dot(C1,P0)
    E[j,:]=E0    #Eq.3-25で計算の場合    
#
invE=np.linalg.inv(E)
invW1=np.linalg.inv(W1) #逆行列を求める
F=np.dot(P0,np.dot(invE,f0))
print("\nF=\n ",F)
# 正解値
#

#calculate x_hat
XH=np.array([[2.], [11.], [-10.]]) #1x3
U=np.array([[1.]]) #入力 
for  k in range(0,knum):
    XX[k]=np.transpose(XH)
    #step1
    #XH0-->XH0[k+1],XH-->XH[k],YH0-->YH0(k+1)
    #入力はこれで良いのだろうか？
    dob[k]=-np.dot(G,X)+G[0,0]*ramp[k] # K0=G[0,0]
    U[0]=dob[k]
    XH0=np.dot(P1,XH)+np.dot(Q1,U) #XH[k]
    YH0=np.dot(C1,XH0) #YH0はスカラとしている    
    #step2
    #XH=XH0+np.dot(F,(y[k]-YH0))
     #XH1-->XH1[k+1],YH0-->YH0(k+1)
    if k!=knum-1: XH=XH0+np.dot(F,(y[k+1]-YH0))
    #if k!=knum-1: XX[k+1]=np.transpose(XH)
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
#ax1.plot(t,YY,'-og') 
#ax1.plot(t,XX[:,0],'-*g') 
ax1.plot(t,y,'-*r') 
ax1.plot(t,yopen,'-*c') 
#ax1.plot(t,ramp,'--b')  #input 
ax1.plot(t,ramp,drawstyle='steps-post',color='b', linestyle='dashed', marker='.')

strg0="重み w={:.3g}".format(w11)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図6-7 LQIプロセス制御の例 :"+strg0, fontname="MS Gothic")

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
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='.')
ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*')
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
#                figure 2                                  #
############################################################
fig = plt.figure(figsize=(8, 8)) # Figureの初期化

#1つの図に様々な大きさのグラフを追加
# https://pystyle.info/matplotlib-grid-sepc/
#縦方向に3つ場所を用意して、2つをss１に、1つをss2用に使う
#
gs = GridSpec(3, 2)  # 縦方向に3つ、横方向に2つの場所を用意
#ss1--> 場所は(0,0)、縦1つ、横１つ、を使用
ss1 = gs.new_subplotspec((0, 0), rowspan=1,colspan=1)  # ax1 を配置する領域
#ss2--> 場所は(2,0)、縦１つ横１つ、を使用
ss2 = gs.new_subplotspec((1, 0), rowspan=1, colspan=1)  # ax2 を配置する領域
ss3 = gs.new_subplotspec((2, 0), rowspan=1, colspan=1)  # ax3 を配置する領域
ss4 = gs.new_subplotspec((0, 1), rowspan=1, colspan=1)  # ax4 を配置する領域
ss5 = gs.new_subplotspec((1, 1), rowspan=1, colspan=1)  # ax5 を配置する領域
ss6 = gs.new_subplotspec((2, 1), rowspan=1, colspan=1)  # ax6 を配置する領域

t=np.arange(0,knum)
# ax1　PLOT
ax1 = plt.subplot(ss1)
ax1.plot(t,XX[:,0],'-*r') 
ax1.plot(t,XX0[:,0],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
strg0=""
plt.title("図6-8 有限整定状態観測(X1)"+strg0, fontname="MS Gothic")

# ax2　PLOT
ax2 = plt.subplot(ss2)
ax2.plot(t,XX[:,1],'-*r') 
ax2.plot(t,XX0[:,1],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図6-8 有限整定状態観測(X2)", fontname="MS Gothic")

# ax3　PLOT
ax3 = plt.subplot(ss3)
ax3.plot(t,XX[:,2],'-*r') 
ax3.plot(t,XX0[:,2],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図6-8 有限整定状態観測(X3)", fontname="MS Gothic")

# ax4　PLOT
ax4 = plt.subplot(ss4)
ax4.plot(t,XX[:,0],'-*r') 
ax4.plot(t,XX0[:,0],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図6-8 有限整定状態観測(X1)", fontname="MS Gothic")

# ax5　PLOT
ax5 = plt.subplot(ss5)
ax5.plot(t,XX[:,1],'-*r') 
ax5.plot(t,XX0[:,1],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図6-8 有限整定状態観測(X2)", fontname="MS Gothic")

# ax6　PLOT
ax6 = plt.subplot(ss6)
ax6.plot(t,XX[:,2],'-*r') 
ax6.plot(t,XX0[:,2],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図6-8 有限整定状態観測(X3)", fontname="MS Gothic")
#

plt.tight_layout()


# 表示
plt.show()    