#digital control 
#デジタル制御　高橋安人

#7.5節 多変数制御系
#p142 図7-7　20240329
#入力u1,u2を選ぶ（line181以降)
#
#また、observerについては記述しない

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")#　=True 指数表記禁止

#---------------------------------------------------
knum=30 #計算step数
ng=5 #gijのデータ数
m=2  #入出力数((u1,u2) (y1,y2))
n=m*ng #P行列の次数

#状態方程式　(4-21)
P=np.zeros((n,n));Q=np.zeros((n,m));C=np.zeros((m,n))
II=np.eye(m,m) #単位行列
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
R=np.array([[p1,0], [0,p2]]) 
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
            P[l+i,l+m+j]=II[i,j]

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

#Get Feedback Gain (7-31)
B1=G1;B2=G2-np.dot(G1,R);B3=G3-np.dot(G2,R)
B4=G4-np.dot(G3,R);B5=G5-np.dot(G4,R)
K0=np.linalg.inv(B1+B2+B3+B4+B5)
K1=np.zeros((m,m))
K2=K0;K3=K0;K4=K0
K5=II+R-np.dot(K0,(G1+G2+G3+G4))

#書籍の値-->発散
#K0=np.array([[5.119,-3.413],[-1.146,3.145]])
#K1=np.zeros((m,m))
#K2=K0;K3=K0;K4=K0
#K5=np.array([[-1.408,0.341],[0.670,-0.143]])

#make Feedback Gain 
for i in range(0,m):
    for j in range(0,m):
        G[i,j]=K0[i,j];G[i,j+m]=K1[i,j];G[i,j+2*m]=K2[i,j]
        G[i,j+3*m]=K3[i,j];G[i,j+4*m]=K4[i,j];G[i,j+5*m]=K5[i,j]

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

#観測器
#Eh--->H(k)の収束状況を確認
CP=np.zeros((m,m*n));CQ=np.zeros((m,m*n)) #temp
#Eobs=np.zeros((m*(n+m),n+m))
#
#観測器用
#YY=np.zeros((knum,3)) #PLOT用array 
#XX0=np.zeros((knum,n)) #PLOT用array
#XX=np.zeros((knum,n)) #PLOT用array
#
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
#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
#Get Feedback Gain
#　(7-31)式により事前に指定する（有限整定）

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
C1=np.array([[1.,0,0,0,0,0,0,0,0,0,0,0],
             [0,1.,0,0,0,0,0,0,0,0,0,0]]) #m*(n+m)

#X=np.zeros((n+m,1))
#入力R(k)をオフセットとして設定するため
Xoff=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

#u[k]にLimitを付ける20240329
u1limit=1.8;u2limit=1.8
u1limit=10;u2limit=10
u0=np.array([[0],[0]])

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
    #--------------------
    u[k]=u0

    #外乱入力はここに入れる？。d(k)に加算?  
    #if k==Noise1:VD[0,0]=1.0  #r1(k)
    #else:VD[0,0]=0
    #if k==Noise2:VD[1,0]=1.0 #r2(k)
    #else:VD[1,0]=0

    X=np.dot(P1,X)+np.dot(Q1,(UD+VD))
    #XX0[k]=np.transpose(X) #for PLOT
    y[k]=np.transpose(np.dot(C1,X))
    #
#######################################################
#                 open loop                           #
#           X=np.dot(P,X)+np.dot(Q,U)                 #     
#######################################################
# feedbackなし、P、Q,Cを使う(P1,Q1,C1ではない)
yopen=np.zeros((knum,m))# PLOT用
uopen=np.zeros((knum,m)) # input
Xopen=np.zeros((n,1)) #状態ベクトルX=np.zeros((n,1)) #状態ベクトル
Uop=np.zeros((m,1))
#

for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    uopen[k]=rinp[k]
    #Uopとuopenの行列構造が違う。よって、要素に値で入力する！！
    Uop[0,0]=uopen[k,0];Uop[1,0]=uopen[k,1] #計算のため置き換え
            
    #外乱入力はここに入れる？。d(k)に加算  
    if k==Noise1:VD[0,0]=1.0  #r1(k)
    else:VD[0,0]=0
    if k==Noise2:VD[1,0]=1.0 #r2(k)
    else:VD[1,0]=0

    Xopen=np.dot(P,Xopen)+np.dot(Q,(Uop+VD))
    #Xopen=np.dot(P,Xopen)+np.dot(Q,Uop)
    yopen[k]=np.transpose(np.dot(C,Xopen))
    #
########################################################################
#            次からは観測器による状態推定                       #
######################################################################## 
#また、observerについては記述しない
#後日きがむけば。。


#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################

############################################################
#                figure 1                                  #
############################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(4.5,6)) # Figureの初期化
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
#ax1.plot(t,y[:,0],'-*r',label="y1(k)") 
#ax1.plot(t,y[:,1],'-*m',label="y2(k)") 
ax1.plot(t,yopen[:,0],'-*r',label="y1_step(k)") 
ax1.plot(t,yopen[:,1],'-*m',label="y2_step(k)") 

ax1.plot(t,rinp[:,0],drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r1(k)")
ax1.plot(t,rinp[:,1],drawstyle='steps-post',color='c', linestyle='dashed', marker='',label="r2(k)")

strg0="u_limit={:.3g} ".format(u1limit)
strg0=""
#strg1="K={:.3g},Kc={:.3g},a={:.3g},b={:.3g},".format(Kgain,Kc,a,b)
#strg2="T={:.3g},Kp={:.3g},p={:.3g},q={:.3g}".format(T,Kp,p,q)
plt.title("図7-6 2変数プロセスのステップ応答 :"+strg0, fontname="MS Gothic")

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
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()

############################################################
#                figure 2                                  #
############################################################
#PLOT 実験データ　gij
#from matplotlib.gridspec import GridSpec
#fig = plt.figure(figsize=(5,5)) # Figureの初期化

#t=np.arange(0,ng)
#####11111111111111111111########
# ax1　PLOT
#####11111111111111111111########
#plt.plot(t,g11,'-*c',label="g11")
#plt.plot(t,g12,'-*r',label="g12") 
#plt.plot(t,g21,'-*m',label="g21") 
#plt.plot(t,g22,'-*b',label="g22") 
#strg0=""
#plt.title("2変数プロセス g(i) :"+strg0, fontname="MS Gothic")

#plt.ylabel("Response ")
#plt.xlabel("step (k)")

#plt.minorticks_on()
#plt.legend(loc='lower right')
#plt.grid() #ax1.grid() でも良い
#
# 表示
plt.show()  


