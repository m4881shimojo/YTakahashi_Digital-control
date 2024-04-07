#digital control 
#デジタル制御　高橋安人
#20240321 shimojo
#p138　図7-3
#
#p118 6.5節　0型プラントのLQI制御を基礎として使う
#p118v1.py
#積分要素を入れるための手法の一般化　u(k)---＞d(k)
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=5,  floatmode="fixed")#　=True 指数表記禁止
#
m=1 # m個のu入力 (今回意味はない)

#以下書籍表7-1の値
#case(a)
n=6;p0=0.708;g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108 #case(a)
#case(b)
#n=6;p0=0.737;g1=0.004;g2=0.112;g3=0.179;g4=0.179;g5=0.155;g6=0.124 #case(b)
#case(c)
#n=5;p0=0.689;g1=0.014;g2=0.144;g3=0.183;g4=0.161;g5=0.124#;g6=0.085 #case(c)
#---------------------------------------------------------------------------------
#p136v1.pyで求めた値
#case(a)
#n=6;p0=0.70788;g1=0.00763;g2=0.12794;g3=0.1827;g4= 0.17224;g5=0.14101;g6=0.10764 #case(a1)
#case(b)
#n=6;p0=0.73687;g1=0.00319;g2=0.102;g3=0.16266;g4=0.16313;g5=0.1407;g6=0.1127 #case(b1)
#case(c)
#n=5;p0=0.6886;g1= 0.01531;g2=0.15946;g3=0.20368;g4=0.17896;g5=0.1378#;g6=0.09935 #case(c1)

#case(d) T1=12;T2=7.2;L1=3.6 shimojo version shimojo version 20%UP
#n=6;p0=0.7619;g1=0.0009;g2= 0.08062;g3= 0.14412;g4= 0.15281;g5= 0.13792;g6= 0.11514 #case(d1)
#case(e) T1=8;T2=4.8;L1=2.4 shimojo version shimojo version 20%Down
#n=5;p0= 0.6477;g1= 0.02797;g2=0.19767;g3= 0.22438;g4= 0.1815;g5=0.12982;g6=0.08732  #case(e1)

#case(f) T1=13;T2=7.8;L1=3.9 shimojo version shimojo version 30%UP
#n=7;p0=0.7723;g1=0.00005;g2= 0.06294;g3=0.12728;g4=0.14207;g5=0.13349;g6=0.11552;g7=0.09534#case(f1)
#case(g)　T1=7;T2=4.2;L1=2.1 shimojo version 30%Down
#n=5;p0=0.5987;g1= 0.04844;g2=0.2435;g3=0.24235;g4=0.17731;g5=0.11573;g6=0.07138 #case(g1)


#----- 以下のコマンド行の下をチェックのこと！！----#　
######　    有限整定制御のため　決め打ち     ######

b1=g1;b2=g2-p0*g1;b3=g3-p0*g2;b4=g4-p0*g3;b5=g5-p0*g4;b6=g6-p0*g5

P=np.array([[0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,g6],
            [0,0,0,0,0,p0]]) 
Q=np.array([[g1],
            [g2],
            [g3],
            [g4],
            [g5],
            [1]]) 
#
C=np.array([[1,0,0,0,0,0]]) 

#
knum=40 #収束計算の回数上限
#今回 m-->入力数の意味と混同する使い方をしている
#

P1=np.zeros((n+1,n+1));Q1=np.zeros((n+1,m))
C1=np.zeros((m,n+1))

H=np.zeros((n+m,n+m));Eh=np.zeros((n+m,n+m))
W0=np.zeros((n+m,n+m));w=np.zeros((m,m))
G=np.zeros((m,n+m));F=np.zeros((n+m,m)) #Gain
C1=np.zeros((m,n+1))


rinp=np.zeros(knum)
X=np.zeros((n+1,1)) #状態変数

#u=np.ones((knum,m))
u=np.zeros((knum,m)) #入力
d=np.ones((knum,m)) #刻み入力
dob=np.ones((knum,m))#observar用

U=np.zeros((m,1))
UD=np.zeros((m,1))
VD=np.zeros((m,1)) #noise
#y=np.zeros((knum,m))


#観測器用
y=np.zeros((knum,m))
E=np.zeros((n+1,n+1))
#YY=np.zeros(knum,n+1) #PLOT用array
XX0=np.zeros((knum,n+1)) #状態変数X　PLOT用array　
XX=np.zeros((knum,n+1)) #状態変数Xの推定値　PLOT用array
#
#
########################################################################
#            次からは P,Q 行列を P1,Q1 行列に変換する                    #
########################################################################
CP=np.dot(C,P);CQ=np.dot(C,Q) #Eq6-24
C1=np.array([[1,0,0,0,0,0,0]]) 
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


##############################
#
#################################################
######　  （リカチ行列の収束計算は不要　）   ######
######　    有限整定制御のため　決め打ち     ######
#################################################

#以下は、T1,T2,Lの20%､30%　UP/Down での値
# コメントアウトして利用する
p0=0.708;g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108 #case(a)
#p0=0.73687;g1=0.00319;g2=0.102;g3=0.16266;g4=0.16313;g5=0.1407;g6=0.1127 #case(b1)
#p0=0.6886;g1= 0.01531;g2=0.15946;g3=0.20368;g4=0.17896;g5=0.1378#;g6=0.09935 #case(c1)

#p0=0.70788;g1=0.00763;g2=0.12794;g3=0.1827;g4= 0.17224;g5=0.14101;g6=0.10764 #case(a1)
#p0=0.7619;g1=0.0009;g2= 0.08062;g3= 0.14412;g4= 0.15281;g5= 0.13792;g6= 0.11514 #case(d1)
#p0= 0.6477;g1= 0.02797;g2=0.19767;g3= 0.22438;g4= 0.1815;g5=0.12982;g6=0.08732  #case(e1)
#p0=0.7723;g1=0.00005;g2= 0.06294;g3=0.12728;g4=0.14207;g5=0.13349;g6=0.11552;g7=0.09534#case(f1)
#p0=0.5987;g1= 0.04844;g2=0.2435;g3=0.24235;g4=0.17731;g5=0.11573;g6=0.07138 #case(g1)
#
b1=g1;b2=g2-p0*g1;b3=g3-p0*g2;b4=g4-p0*g3;b5=g5-p0*g4;b6=g6-p0*g5

K0=1/(b1+b2+b3+b4+b5+b6)
G[0,0]=K0;G[0,1]=0.0;G[0,2]=K0;G[0,3]=K0;G[0,4]=K0;G[0,5]=K0
G[0,6]=1+p0-K0*(g1+g2+g3+g4+g5)#;G[0,6]=-3.518#書籍おかしい
print("Gain= ",G)


#################################################
######　    ここから応答の計算     　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
# 
#　ステップ＋ランプ入力
Noise_start=25 # #指令値を一回のみ変える。外乱とは言わない
ramp_start=500;ramp_end=550

#
for i in range(0,knum):
    if i<ramp_start: rinp[i]=1.0 #step input
    elif i>=ramp_start and i<=ramp_end: rinp[i]=1.0+(i-ramp_start)*0.5 #rump input
    #上記0.5は任意。ランプのスロープを調整する
    else:rinp[i]=1.0
 

#　以下は外乱？としての初期値変化
#状態Xの初期値を与える。今回はn=3。
X0=np.array([[0],[0],[0],[0],[0],[0],[0]]) # 初期値
X=X+X0

#rinp(k)用 n=7
Xoff=np.array([[0.0],[0.0],[0.0],[0],[0],[0],[0]]) 
ulimit=1.8 #制御入力に制限を付ける

y[0]=0.;d[0]=0
for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0; Xoff[2,0]=0.0
    d[k]=-np.dot(G,(X-Xoff))
    #
    UD[0]=d[k] #計算のため置き換え
    if k==0:u[k]=d[k]# For Plot
    #else:u[k]=u[k-1]+d[k]
    else:u0=u[k-1]+d[k] #u[k]にLimitを付けるため

    #u[k]にLimitを付ける20240323
    if u0>ulimit: u0=ulimit
    elif u0<-ulimit: u0=-ulimit
    u[k]=u0
    #
    #外乱入力はここに入れる？
    if k==Noise_start:VD[0]=0.5 #１刻みの差分入力のため
    else:VD[0]=0

    X=np.dot(P1,X)+np.dot(Q1,(UD+VD))
    XX0[k]=np.transpose(X) #for PLOT 状態変数X
    #
    y[k]=np.dot(C1,X)
    #X=P.dot(X)+Q.dot(U)
#######################################################
#                 open loop                           #
#######################################################
yopen=np.zeros((knum,m))# PLOT用
uopen=np.zeros(knum) # input
Xopen=np.zeros((n,1)) #状態ベクトルX=np.zeros((n,1)) #状態ベクトル
Uop=np.zeros((m,1))
#yopen[0]=0.;uopen[0]=0

for k in range(1,knum): #
    #d[k] 積分が入ってると、こうなるのかな？
    uopen[k]=rinp[k]
    Uop[0]=uopen[k] #計算のため置き換え
        
    #外乱入力はここに入れる？
    if k==Noise_start:VD[0]=0.5 #１刻みの差分入力のため
    else:VD[0]=0 

    Xopen=np.dot(P,Xopen)+np.dot(Q,(Uop+VD))
    yopen[k]=np.dot(C,Xopen)
    #
#        
########################################################################
#             次からは観測器による状態推定    　　                      #
# 有限整定制御のため決め打ち --＞このため、p0は有限整定制御のところ要確認 #  
######################################################################## 
x_hat=np.zeros((knum,n+1))
#x_hat0=np.zeros((knum,n))
XH=np.zeros((n+1,1)) #状態ベクトルx_hat
XH0=np.zeros((n+1,1)) #状態ベクトルx_hat0

#Calculate E（可観測行列 Eq.3-24）
#F(P)=P^n+α1P(n-1)＋．．．αnI
#F(P)=P^n　（有限整定観測器、αn=0）とした#

F[0,0]=1;F[1,0]=p0;F[2,0]=p0**2;F[3,0]=p0**3;F[4,0]=p0**4
F[5,0]=p0**5;F[6,0]=p0**6
print("\nF=\n ",F)
# 正解値
#

#calculate x_hat
XH=np.array([[0.],[0.],[0.],[0.],[0.],[0.],[0.]]) #1x3
U=np.array([[1.]]) #入力 
for  k in range(0,knum):
    XX[k]=np.transpose(XH)
    #step1
    #XH0-->XH0[k+1],XH-->XH[k],YH0-->YH0(k+1)
    #入力はこれで良いのだろうか？
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0; Xoff[2,0]=0.0
    dob[k]=-np.dot(G,(X-Xoff))
    #dob[k]=-np.dot(G,X)+G[0,0]*rinp[k] # K0=G[0,0]

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
############################################################
#                figure 1                                  #
############################################################
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
ax1.plot(t,yopen,'-*c',label="yopen(k)") 
ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)")

strg0="重み"
plt.title("図6-7 LQIプロセス制御の例 :"+strg0, fontname="MS Gothic")

#Ymax=np.amax(y); Ymin=0.0
strg1=" Gain: , {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}".format(G[0,0],G[0,1],G[0,2],G[0,3],G[0,4],G[0,5],G[0,6])
strg2=" FB  : {:.5g}, {:.5g}, {:.5g},..p(^(n-1))......".format(F[0,0],F[1,0],F[2,0])

Ymax=1.6; Ymin=-0.1
xp=knum*2/10; yp=Ymax*4/10  #plt.textの位置座標
plt.text(xp,yp, strg1 ) #
plt.text(xp,yp-Ymax*1/10, strg2, fontname="MS Gothic" ) 

#plt.text(xp,yp, strg1 ) #
#
#
#plt.xlim(0,knum)
plt.ylim(-0.05,1.2)
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
plt.ylabel("Response ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
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