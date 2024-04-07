#digital control 
#デジタル制御　高橋安人
#7.5節 多変数制御系
#p142 図7-7　20240329
#observerについて記述した20240406

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")#　=True 指数表記禁止

#---------------------------------------------------
knum=50 #計算step数
ng=5 #gijのデータ数
m=2  #入出力数((u1,u2) (y1,y2))
n=m*ng #P行列の次数

#u[k]にLimitを付ける20240329
u1limit=1.8;u2limit=1.8
#u1limit=1000;u2limit=1000

#状態方程式　(4-21)
P=np.zeros((n,n));Q=np.zeros((n,m));C=np.zeros((m,n))
Im=np.eye(m,m) #単位行列

#----------------------------------------------------
#応答の計算のためのarray
rinp=np.zeros((knum,m)) #r(k)

X=np.zeros((n+m,1)) #状態変数
u=np.zeros((knum,m)) #入力
d=np.ones((knum,m)) #刻み入力
#
UD=np.zeros((m,1))
VD=np.zeros((m,1)) #noise
y=np.zeros((knum,m))#PLOT

#観測器用
XX0=np.zeros((knum,n+m)) #状態変数X　PLOT用array　
XX=np.zeros((knum,n+m)) #状態変数Xの推定値　PLOT用array
yy0=np.zeros((m,1))

x_hat=np.zeros((knum,n+1))
XH=np.zeros((n+m,1)) #状態ベクトルx_hat
XH0=np.zeros((n+m,1)) #状態ベクトルx_hat0

#make Gi (4-21)
G1=np.zeros((m,m));G2=np.zeros((m,m));G3=np.zeros((m,m))
G4=np.zeros((m,m));G5=np.zeros((m,m))
R=np.zeros((m,m)) #pi (4-21)


#以下書籍表7-2の値(実験値の応答データ)
g11=np.zeros((ng));g12=np.zeros((ng));g21=np.zeros((ng));g22=np.zeros((ng))
p1=0.8;p2=0.6
#For input u1: 
#u1-->y1
g11[0]=0.1;g11[1]=0.21;g11[2]=0.18;g11[3]=0.15;g11[4]=0.13
#u1-->y2
g12[0]=-0.05;g12[1]=-0.05;g12[2]=0.01;g12[3]=0.1;g12[4]=0.09
#For input u2: 
#u2-->y1
g21[0]=0.02;g21[1]=0.06;g21[2]=0.14;g21[3]=0.18;g21[4]=0.12
#u2-->y2
g22[0]=0.08;g22[1]=0.2;g22[2]=0.23;g22[3]=0.19;g22[4]=0.14

# (4-21)
# R matrix
R=np.array([[p1,0], [0,p2]]) 
R=np.array([[p1,0.1], [-0.1,p2]]) #試しに与えた（根拠なし）

#Gi matrix
G1=np.array([[g11[0],g12[0]],[g21[0],g22[0]]])
G2=np.array([[g11[1],g12[1]],[g21[1],g22[1]]])
G3=np.array([[g11[2],g12[2]],[g21[2],g22[2]]])
G4=np.array([[g11[3],g12[3]],[g21[3],g22[3]]])
G5=np.array([[g11[4],g12[4]],[g21[4],g22[4]]])

#Make P matrix (4-21)
#write I
for l in range(0,ng):
    for i in range(0,m):
        for j in range(0,m):
            P[l+i,l+m+j]=Im[i,j]

#write Gn
for i in range(0,m):
    for j in range(0,m):
        P[(m*(ng-2)+i),(m*ng-2+j)]=G5[i,j]
        P[(m*(ng-1)+i),(m*ng-2+j)]=R[i,j]
        
#Make Q marix
#直接作ってしまった Giから作りたかったが。。
Q=np.array([[g11[0],g12[0]],[g21[0],g22[0]],
              [g11[1],g12[1]],[g21[1],g22[1]],
              [g11[2],g12[2]],[g21[2],g22[2]],
              [g11[3],g12[3]],[g21[3],g22[3]],
              [1.0,0.0],[0.0,1.0]])
#Make C marix　(get y1,y2)
C=np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.]])

P1=np.zeros((n+m,n+m));Q1=np.zeros((n+m,m))
G=np.zeros((m,n+m)) #Feedback Gain
C1=np.zeros((m,n+1))

##########################################################
#　      (7-31)式により事前に指定する（有限整定）          #
#         Get Feedback Gain (7-31)                       #
##########################################################
B1=G1;B2=G2-np.dot(G1,R);B3=G3-np.dot(G2,R)
B4=G4-np.dot(G3,R);B5=G5-np.dot(G4,R)
K0=np.linalg.inv(B1+B2+B3+B4+B5)
K1=np.zeros((m,m))
K2=K0;K3=K0;K4=K0
K5=Im+R-np.dot(K0,(G1+G2+G3+G4))
print("\nK0=\n",K0)
print("K5=\n",K5)

In=np.eye(n,n)
invIP=np.linalg.inv((In-P))
dcGain=np.dot(C,np.dot(invIP,Q))
print("dc_Gain=\n",dcGain)

#make Feedback Gain 
for i in range(0,m):
    for j in range(0,m):
        G[i,j]=K0[i,j];G[i,j+m]=K1[i,j];G[i,j+2*m]=K2[i,j]
        G[i,j+3*m]=K3[i,j];G[i,j+4*m]=K4[i,j];G[i,j+5*m]=K5[i,j]


########################################################################
#            次からは P,Q 行列を P1,Q1 行列に変換する                    #
########################################################################
#
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
print("\nR matrix\n",R)
#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
#Get Feedback Gain
#　(7-31)式により事前に指定する（有限整定）
#  これ以前のところで計算済
#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2...] 計算済み    ######
#################################################
# 
#　ステップ＋ランプ入力
Sup1=30;Sup2=40
Noise1=10;Noise2=20
#
#今回はstep状変化とする
for i in range(0,knum):
    rinp[i,0]=3.0;rinp[i,1]=1.0
    #rinp[i,0]=0.0;rinp[i,1]=1.0
    if i>Sup1: rinp[i,1]=2.0 #u2 step_UP
    if i>Sup2: rinp[i,0]=4.0 #u1 step_UP
    
#状態Xの初期値を与える。
#X0=np.array([[0],...,[0],[0]]) #n+m
#
C1=np.array([[1.,0,0,0,0,0,0,0,0,0,0,0],
             [0,1.,0,0,0,0,0,0,0,0,0,0]]) #m*(n+m)

#X=np.zeros((n+m,1))
#入力R(k)をオフセットとして設定するため
Xoff=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
u0=np.array([[0],[0]])#Temp use. uLimitを入れるため

for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    #入力R(k)をオフセットとして設定
    Xoff[0,0]=rinp[k,0]; Xoff[1,0]=rinp[k,1]
    UD=-np.dot(G,(X-Xoff)) 
    d[k]=UD.T #計算のため置き換え
    if k==0:u[k]=d[k]# For Plot
    #else:u[k]=u[k-1]+d[k]
    else: u0=u[k-1]+d[k]

     #--------------------
    #u[k]にLimitを付ける20240329
    if u0[0]>u1limit: u0[0]=u1limit
    elif u0[0]<-u1limit: u0[0]=-u1limit
    if u0[1]>u2limit: u0[1]=u2limit
    elif u0[1]<-u2limit: u0[1]=-u2limit
    u[k]=u0
    #--------------------

    #外乱入力はここに入れる？。d(k)に加算?  
    if k==Noise1:VD[0,0]=1.0  #r1(k)
    else:VD[0,0]=0
    if k==Noise2:VD[1,0]=1.0 #r2(k)
    else:VD[1,0]=0

    X=np.dot(P1,X)+np.dot(Q1,(UD+VD))
    XX0[k]=np.transpose(X) #for PLOT 状態変数X
    y[k]=np.transpose(np.dot(C1,X))
    #

########################################################################
#            次からは観測器による状態推定                       #
######################################################################## 
#
#Calculate F（可観測行列 Eq.4-30）
F=np.zeros((n+m,m)) #Observer Feedback Gain
Im=np.eye(m,m)
F0=Im;F1=R;F2=np.dot(R,F1);F3=np.dot(R,F2);F4=np.dot(R,F3)
F5=np.dot(R,np.linalg.inv((G5)))
#make Observer Feedback Gain 
for i in range(0,m):
    for j in range(0,m):
        F[i,j]=F0[i,j];F[i+m,j]=F1[i,j];F[i+2*m,j]=F2[i,j]
        F[i+3*m,j]=F3[i,j];F[i+4*m,j]=F4[i,j];F[i+5*m,j]=F5[i,j]

print("\nF=\n ",F)
# 
#calculate x_hat
#　わざと初期値に誤差を与える
XH=np.array([[10.],[10.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]) #
XH=np.array([[4.],[2.],[-5.],[5.],[-10.],[-15.],[10.],[20.],[-9.],[8.],[0.],[0.]]) #

for  k in range(0,knum):
    XX[k,:]=np.transpose(XH)
    #step1
    #XH0-->XH0[k+1],XH-->XH[k],YH0-->YH0(k+1)
    #入力はこれで良いのだろうか？
    Xoff[0,0]=rinp[k,0]; Xoff[1,0]=rinp[k,1]
    UD=-np.dot(G,(X-Xoff)) 
    #UDは応答と兼用　あとで問題になるか？
    
    XH0=np.dot(P1,XH)+np.dot(Q1,UD) #XH[k]
    YH0=np.dot(C1,XH0) #

    #step2
    #XH=XH0+np.dot(F,(y[k]-YH0))
     #XH1-->XH1[k+1],YH0-->YH0(k+1)
    if k!=knum-1:
        yy0[0,0]=y[k+1,0]; yy0[1,0]=y[k+1,1]
    #if k!=knum-1: XH=XH0+np.dot(F,(y[k+1]-YH0)) #(4-24)
    XH=XH0+np.dot(F,(yy0-YH0))
    # 

#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#########################################################
#                figure 1                                #
#########################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(7,7)) # Figureの初期化
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
ax1.plot(t,y[:,0],'-*r',label="y1(k)") 
ax1.plot(t,y[:,1],'-*m',label="y2(k)") 
ax1.plot(t,rinp[:,0],drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r1(k)")
ax1.plot(t,rinp[:,1],drawstyle='steps-post',color='c', linestyle='dashed', marker='',label="r2(k)")

strg0="u_limit={:.3g} ".format(u1limit)
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図7-7 2変数プロセスの有限整定制御 :"+strg0, fontname="MS Gothic")

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
#ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*')
plt.ylabel("input")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()

############################################################
#                figure 2                                  #
############################################################
fig = plt.figure(figsize=(8,8)) # Figureの初期化

#1つの図に様々な大きさのグラフを追加
# https://pystyle.info/matplotlib-grid-sepc/
#縦方向に3つ場所を用意して、2つをss１に、1つをss2用に使う
#
gs = GridSpec(3, 2)  # 縦方向に3つ、横方向に2つの場所を用意
#ss1--> 場所は(0,0)、縦1つ、横１つ、を使用
ss1 = gs.new_subplotspec((0, 0), rowspan=1,colspan=1)  # ax1 を配置する領域
#ss2--> 場所は(2,0)、縦１つ横１つ、を使用
#　n+m（今回は12）だけ状態があるが、ここでは表示は6個とした
ss2 = gs.new_subplotspec((1, 0), rowspan=1, colspan=1)  # ax2 を配置する領域
ss3 = gs.new_subplotspec((2, 0), rowspan=1, colspan=1)  # ax3 を配置する領域
ss4 = gs.new_subplotspec((0, 1), rowspan=1, colspan=1)  # ax4 を配置する領域
ss5 = gs.new_subplotspec((1, 1), rowspan=1, colspan=1)  # ax5 を配置する領域
ss6 = gs.new_subplotspec((2, 1), rowspan=1, colspan=1)  # ax6 を配置する領域


t=np.arange(0,knum)
# ax1　PLOT
ax1 = plt.subplot(ss1)
ax1.plot(t,XX0[:,0],'-*r',label="x1") #状態変数X
ax1.plot(t,XX[:,0],'--b',label="x01") #状態変数推定値XH

plt.ylabel("Response ")
plt.xlabel("step (k)")
strg0=""
plt.title("図 有限整定状態観測(X1)"+strg0, fontname="MS Gothic")
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

# ax2　PLOT
ax2 = plt.subplot(ss2)
ax2.plot(t,XX0[:,1],'-*r',label="x2")
ax2.plot(t,XX[:,1],'--b',label="x02")  
plt.ylabel("Response ")
plt.xlabel("step (k)")
plt.title("図 有限整定状態観測(X2)", fontname="MS Gothic")
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

# ax3　PLOT
ax3 = plt.subplot(ss3)
ax3.plot(t,XX0[:,2],'-*r',label="x3")
ax3.plot(t,XX[:,2],'--b',label="x03")  
plt.ylabel("Response ")
plt.xlabel("step (k)")
plt.title("図 有限整定状態観測(X3)", fontname="MS Gothic")
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

# ax4　PLOT  
ax4 = plt.subplot(ss4)
ax4.plot(t,XX0[:,3],'-*r',label="x4") 
ax4.plot(t,XX[:,3],'--b',label="x04")
plt.ylabel("Response ")
plt.xlabel("step (k)")
plt.title("図 有限整定状態観測(X4)", fontname="MS Gothic")
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

# ax5　PLOT
ax5 = plt.subplot(ss5)
ax5.plot(t,XX0[:,4],'-*r',label="x5") 
ax5.plot(t,XX[:,4],'--b',label="x05") 
plt.ylabel("Response ")
plt.xlabel("step (k)")
plt.title("図 有限整定状態観測(X5)", fontname="MS Gothic")
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

# ax6　PLOT
ax6 = plt.subplot(ss6)
ax6.plot(t,XX0[:,5],'-*r',label="x6") 
ax6.plot(t,XX[:,5],'--b',label="x06") 
plt.ylabel("Response ")
plt.xlabel("step (k)")
plt.title("図 有限整定状態観測(X6)", fontname="MS Gothic")
plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い
#

plt.tight_layout()
# 表示
plt.show()  


