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
knum=40 #サンプリング総数
n=6 #次数
m=1 #入力数（今回は1入力のみとしている）
Tsample=0.5 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n))
P=np.eye(n,n);Q0=np.eye(n,n)#
Dpq=np.eye(n,n)# while Loopの中で使うからD=np.eye(n,n)絶対
E=np.zeros((n,n));F=np.zeros((n,m))#


#観測器用
y=np.zeros(knum) #出力の時系列
XX0=np.zeros((knum,n)) #状態ベクトルXの時系列
XX=np.zeros((knum,n)) #Xの推定値XHの時系列(4-21)　

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
#p76　1x6
C=np.array([[1.,0.,0.,0.,0.,0.]]) #m1の変位を出力とする

########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                 #
#    P=np.eye(n,n);Q0=np.eye(n,n);Dpq=np.eye(n,n) #<--MUST             #
######################################################################## 
#calculate P,Q matrix
#list4-1 Pとqの算定
k=0
e0=1.0E-8;e1=1.0 #誤差許容値
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
print("\n -------------P and Q matrix------------")
#print("number of recursion=",k)
print("\nP matrix\n",P)
print("\nQ matrix\n",Q)
#end

########################################################################
#            次からは  P,Q 行列を使って応答を求めます　                  #
########################################################################
#
X=np.zeros((n,1)) #状態ベクトル
U=np.array([[1.]]) #入力 step入力
#　以下は外乱？としての初期値
X0=np.array([[5],[2],[7],[5],[3],[8]])#書籍での例 p77
#X0=np.array([[1],[0],[0],[0],[0],[0]])#Initial condition(外乱？)
#X0=np.array([[0],[0],[0],[0],[0],[0]])#Initial condition

print("\n状態量の初期値\n",X0)
# list 3-2 page42

X=X+X0 #X0　<--初期値
for k in range(0,knum):
    y[k]=np.transpose(np.dot(C,X)) #出力の時系列
    XX0[k]=np.transpose(X) #for PLOT　状態ベクトルXの時系列
    X=np.dot(P,X)+np.dot(Q,U) #「最後にある」と「始めにある」結果の違い 
#        
########################################################################
#            次からは観測器による状態推定                      　　　　 #
######################################################################## 
x_hat=np.zeros((knum,n))
#x_hat0=np.zeros((knum,n))
XH=np.zeros((n,1)) #状態ベクトルx_hat (4-24)
XH0=np.zeros((n,1)) #状態ベクトルx_hat0 (4-24)

##################################
#Calculate E（可観測行列 Eq.3-24）
##################################
#F(P)=P^n+α1P(n-1)＋．．．αnI
#F(P)=P^n　（有限整定観測器、αn=0）とした
#
C1=np.array([[0.], [0], [0.], [0.], [0.], [1.]]) #1x6
C0=np.array((m,n)) # cP(^j)temporary use
P0=np.eye(n,n) # 単位行列----> P(^j)    
for j in range(0,n):
    P0=np.dot(P0,P) # P(^j) 
    C0=np.dot(C,P0) # cP(^j)
    E[j,:]=C0    #Eq.4-25で計算! <-- 重要。間違えないように   
#
invE=np.linalg.inv(E)
F=np.dot(P0,np.dot(invE,C1)) #(4-25)
print("\nF=\n ",F)
# 正解値
#F=np.array([[1.],[4.035],[10.958],[13.520],[8.769],[-11.847]])

#calculate x_hat
#XX-->(knum,6),y0-->(knum,)
#XH,XH0,Q,F--->(6, 1)
#YH0,U--->(1, 1),C --->(1, 6)

#状態変数の初期値plt.grid()
XH=np.array([[0.], [0], [0.], [0.], [0.], [0.]]) #1x6
U=np.array([[1.]]) #入力 入力 step入力
for  k in range(0,knum):
    XX[k]=np.transpose(XH) #XHの時系列(4-21)
    ####################################
    #step1
    ####################################
    #XH0-->XH0[k+1],XH-->XH[k],YH0-->YH0(k+1)
    XH0=np.dot(P,XH)+np.dot(Q,U) #XH[k] (4-24)
    YH0=np.dot(C,XH0) #YH0はスカラとしている
    ####################################    
    #step2
    ####################################
    #XH=XH0+np.dot(F,(y[k]-YH0))
     #XH1-->XH1[k+1],YH0-->YH0(k+1)
    if k!=knum-1: XH=XH0+np.dot(F,(y[k+1]-YH0)) #(4-24)
    #if k!=knum-1: XX[k+1]=np.transpose(XH)
    # 
#
########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
#plt.clf()
#plt.close()
############################################################
#                figure 1                                  #
############################################################
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
ax1.plot(t,XX[:,0],'-*r',label="x0_hat") 
ax1.plot(t,XX0[:,0],'--b',label="x0") 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
strg0=""
plt.title("図 表4-6 有限整定状態観測(X1)"+strg0, fontname="MS Gothic")
plt.legend(loc='upper right')
plt.grid()

# ax2　PLOT
ax2 = plt.subplot(ss2)
ax2.plot(t,XX[:,1],'-*r',label="x1_hat") 
ax2.plot(t,XX0[:,1],'--b',label="x1") 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X2)", fontname="MS Gothic")
plt.legend(loc='upper right')
plt.grid()

# ax3　PLOT
ax3 = plt.subplot(ss3)
ax3.plot(t,XX[:,2],'-*r',label="x2_hat") 
ax3.plot(t,XX0[:,2],'--b',label="x2") 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X3)", fontname="MS Gothic")
plt.legend(loc='upper right')
plt.grid()

# ax4　PLOT
ax4 = plt.subplot(ss4)
ax4.plot(t,XX[:,3],'-*r',label="x3_hat") 
ax4.plot(t,XX0[:,3],'--b',label="x3") 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X4)", fontname="MS Gothic")
plt.legend(loc='upper right')
plt.grid()

# ax5　PLOT
ax5 = plt.subplot(ss5)
ax5.plot(t,XX[:,4],'-*r',label="x4_hat") 
ax5.plot(t,XX0[:,4],'--b',label="x4") 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X5)", fontname="MS Gothic")
plt.legend(loc='upper right')
plt.grid()

# ax6　PLOT
ax6 = plt.subplot(ss6)
ax6.plot(t,XX[:,5],'-*r',label="x5_hat") 
ax6.plot(t,XX0[:,5],'--b',label="x5") 
plt.ylabel("Responce ")
plt.xlabel("step (k)")
plt.title("図 表4-6 有限整定状態観測(X6)", fontname="MS Gothic")
plt.legend(loc='upper right')
plt.grid()
#

plt.tight_layout()

############################################################
#                figure 2                                  #
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
#ax1.plot(t,XX[:,0],'-*r')
ax1.plot(t,y,'-*r',label="y(k)")  
ax1.plot(t,XX[:,1],'-*b',label="x1(k)")
#ax1.plot(t,XX[:,2],'-*r')
#ax1.plot(t,ramp,'--b')  #input 
#ax1.plot(t,ramp,drawstyle='steps-post',color='b', linestyle='dashed', marker='.')

strg0="状態変数初期値="+str(np.transpose(X0))
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図 3連振動系の有限整定状態観測 :"+strg0, fontname="MS Gothic")
plt.legend(loc='upper right') #labelの表示
#Ymax=np.amax(y); Ymin=0.0
#xp=knum*3/10; yp=Ymax*8/10  #plt.textの位置座標
#strg1=" Gain: K1={:.5g},K2={:.5g}".format(G[0,0],G[0,1])
#strg1=" Gain: "
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

# 表示
plt.show()   

