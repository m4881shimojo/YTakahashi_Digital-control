#digital control 
#デジタル制御　高橋安人
#
#4.5節　観測器による状態推定
#p77表6　#20231218
#観測器状態推定FBをもとめる
#
#制御対象--->　p57ver2 3連振動系
#p239　｢例７｣図A-3(d)の3連振動系を例として利用する
#　入力はステップ入力とした

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
knum=80 #サンプリング総数
n=6 #次数
m=1 #入力数（今回は1入力のみとしている）
T=0.5 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n))
P=np.eye(n,n);Q0=np.eye(n,n)#
Dpq=np.eye(n,n)# while Loopの中で使うからD=np.eye(n,n)絶対

E=np.zeros((n,n));F=np.zeros((n,m))#
W=np.zeros(n);Z=np.zeros(n)

#観測器用
y=np.zeros(knum)
YY=np.zeros(knum) #PLOT用array
XX0=np.zeros((knum,n)) #PLOT用array
XX=np.zeros((knum,n)) #PLOT用array

# #https://analytics-note.xyz/programming/numpy-printoptions/
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止

#三連振動系
#4.1　プラント微分方程式と等価の差分式を参照のこと
A=np.array([[0,1.0,0.,0.,0.,0.],
            [-k1,-b1,k1,b1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [k1,b1,-(k1+k2),-(b1+b2),k2,b2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,k2,b2,-(k2+k3),-(b2+b3)]])
#p76　6x1
B=np.array([[0.],
            [1.], #<--外力をm1に加える
            [0.],
            [0.],
            [0.],
            [0.]]) 
#76　1x6
C=np.array([[1.,0.,0.,0.,0.,0.]]) #m1の変位を出力とする

########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                 #
#    P=np.eye(n,n);Q0=np.eye(n,n);Dpq=np.eye(n,n) #<--MUST             #
######################################################################## 
#calculate P,Q matrix
#list4-1 Pとqの算定
k=0
e0=0.000001 #1.0E-6
e1=1.0
A=T*A
while e1>e0:
    k=k+1
    #Dpq,Epq working matrix
    Dpq=np.dot(A,Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))                                                                                                                                                                                                                                                                                                                                                                  
    
#
# get Q matrix
Q0=T*Q0; Q=np.dot(Q0,B)            
#calculate end
# P and Q matrix was calculated

# check calcutated matric P,Q
# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
print("number of recursion=",k)
print("\nP matrix")
print(P)
print("\nQ matrix")
print(Q)
#end

########################################################################
#            次からは  P,Q 行列を使って応答を求めます　                  #
########################################################################
#
X=np.zeros((n,1)) #状態ベクトル
U=np.array([[1.]]) #入力 step入力
#　以下は外乱？としての初期値
X0=np.array([[5],[2],[7],[5],[3],[6]])#書籍での例
#X0=np.array([[1],[0],[0],[0],[0],[0]])#Initial condition(外乱？)
#X0=np.array([[0],[0],[0],[0],[0],[0]])#Initial condition

print("\n状態量の初期値\n",X0)
# list 3-2 page42

X=X+X0
for k in range(0,knum):
    #X=np.dot(P,X)+np.dot(Q,U)
    y[k]=np.transpose(np.dot(C,X))
    YY[k]=np.transpose(np.dot(C,X)) #for PLOT
    XX0[k]=np.transpose(X) #for PLOT
    X=np.dot(P,X)+np.dot(Q,U)
       
#        
########################################################################
#            次からは観測器による状態推定                       #
######################################################################## 
x_hat=np.zeros((knum,n))
#x_hat0=np.zeros((knum,n))
XH=np.zeros((n,1)) #状態ベクトルx_hat
XH0=np.zeros((n,1)) #状態ベクトルx_hat0

#Calculate E（可観測行列 Eq.3-24）
#F(P)=P^n+α1P(n-1)＋．．．αnI
#F(P)=P^n　（有限整定観測器、αn=0）とした
#
C1=np.array([[0.], [0], [0.], [0.], [0.], [1.]]) #1x6
C0=np.array((m,n)) 
P0=np.eye(n,n)      
for j in range(0,n):
    P0=np.dot(P0,P)
    C0=np.dot(C,P0)
    E[j,:]=C0    #Eq.3-25で計算の場合    
#
invE=LA.inv(E)
F=np.dot(P0,np.dot(invE,C1))
print("\nF=\n ",F)
# 正解値
#F=np.array([[1.],[4.035],[10.958],[13.520],[8.769],[-11.847]])

#calculate x_hat
XH=np.array([[0.], [0], [0.], [0.], [0.], [0.]]) #1x6
U=np.array([[1.]]) #入力 入力 step入力
for  k in range(0,knum):
    XX[k]=np.transpose(XH)
    #step1
    #XH0-->XH0[k+1],XH-->XH[k],YH0-->YH0(k+1)
    XH0=np.dot(P,XH)+np.dot(Q,U) #XH[k]
    YH0=np.dot(C,XH0) #YH0はスカラとしている    
    #step2
    #XH=XH0+np.dot(F,(y[k]-YH0))
     #XH1-->XH1[k+1],YH0-->YH0(k+1)
    if k!=knum-1: XH=XH0+np.dot(F,(y[k+1]-YH0))
    #if k!=knum-1: XX[k+1]=np.transpose(XH)
    # 
#
########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
#plt.clf()
#plt.close()
from matplotlib.gridspec import GridSpec
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
plt.title("図 表4-6 有限整定状態観測(X1)"+strg0, fontname="MS Gothic")

# ax2　PLOT
ax2 = plt.subplot(ss2)
ax2.plot(t,XX[:,1],'-*r') 
ax2.plot(t,XX0[:,1],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X2)", fontname="MS Gothic")

# ax3　PLOT
ax3 = plt.subplot(ss3)
ax3.plot(t,XX[:,2],'-*r') 
ax3.plot(t,XX0[:,2],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X3)", fontname="MS Gothic")

# ax4　PLOT
ax4 = plt.subplot(ss4)
ax4.plot(t,XX[:,3],'-*r') 
ax4.plot(t,XX0[:,3],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X4)", fontname="MS Gothic")

# ax5　PLOT
ax5 = plt.subplot(ss5)
ax5.plot(t,XX[:,4],'-*r') 
ax5.plot(t,XX0[:,4],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X5)", fontname="MS Gothic")

# ax6　PLOT
ax6 = plt.subplot(ss6)
ax6.plot(t,XX[:,5],'-*r') 
ax6.plot(t,XX0[:,5],'--b') 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X6)", fontname="MS Gothic")
#

plt.tight_layout()
# 表示
plt.show()   

