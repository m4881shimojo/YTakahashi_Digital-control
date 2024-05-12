#digital control 
#デジタル制御　高橋安人
#20240430 shimojo
#p166 Fig 8-7
#同定および適応制御
#8.4　模型追従適応制御
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=2 #3次系
nump=4 #W,F
m=1 # m個のu入力
knum=130 #サンプル数
#
P=np.zeros((n,n));Q=np.zeros((n,m)) #パラメータが不変のプラント
rinp=np.zeros(knum)#ramp入力

X=np.zeros((n,1));X0=np.zeros((n,1)) #MRAC
y=np.zeros((knum,1)) #MRAC

u=np.zeros((knum,m)) #MRAC
U=np.zeros((m,1)) #MRAC
V=np.zeros((m,1)) #MRAC & 有限整定
Plot1=np.zeros(knum) # for Plot
Plot2=np.zeros(knum) # for Plot
Plot3=np.ones(knum) # for Plot 補助線

#--------------------------------------
# パラメータが不変のプラント
#　List 8-2では使わない
Tsample=0.1 #sampling period
p=np.exp(-Tsample)
#
P=np.array([[1,1-p],
            [0,p]]) #Eq.6-16
Q=np.array([[p+Tsample-1],
            [1-p]])
C=np.array([[1,0]])
#--------------------------------------

# page 167 list 8-2 MRAC test
#パラメータが変化するプラント
P0=np.zeros((n,n));Q0=np.zeros((n,m)) 
W=np.zeros((nump,m));W0=np.zeros((nump-1,m)) #縦V
F=np.zeros((nump,m));F0=np.zeros((nump-1,m)) #縦V
#F1=np.zeros((nump,m))
B=np.zeros((nump-1,m)) #縦V
R=np.zeros((nump-1,m)) #縦V
G=0.1 #Gain given
d1=0.2 # 書籍のd1=0.2だと発振？
amplitude=0.25#b(粘弾性係数)振幅　<---制御結果へ影響敏感
y0=0.0 #Plant 出力

#-------MRAC parameter-----------
b=1;p=np.exp(-b*Tsample) #b 粘弾性係数
c0=(p+b*Tsample-1)/b**2;c1=(1-p-b*Tsample)/b**2 #fig 8-6
q0=c0/(1-p);q1=c1/(1-p) #fig 8-6
b0=(p+b*Tsample-1)/(b*Tsample*(1-p));b1=(1-p-b*Tsample)/(b*Tsample*(1-p)) #Eq.8-22
F[0,0]=c0;F[1,0]=c1;F[2,0]=d1+1+p;F[3,0]=-p #Eq.8-33
B[0,0]=b0;B[1,0]=b1+d1*b0;B[2,0]=d1*b1 #Eq.8-30
F0[0,0]=c1;F0[1,0]=d1+1+p;F0[2,0]=-p #Eq.8-32
#---------------------------------
#-------有限整定制御 parameter-----
U1=np.zeros((m,1))
X1=np.zeros((n,1))
y1=np.zeros((knum,1))
K0=b/(Tsample*(1-p)) #Eq.8-21
K1=(1-p-b*(p**2)*Tsample)/(Tsample*(1-p)**2) #Eq.8-21
Gk=np.array([[K0,K1]]) #有限整定Gain
Xoff=np.array([[0.0],[0.0]])#
#---------------------------------

#--------------　ステップ入力--------------
for i in range(0,knum):
    if i <20 :rinp[i]=5.0*i
    #elif i>=75 :rinp[i]=100.0
    elif i>20 and i <=40: rinp[i]=100
    elif i>40 and i <=60: rinp[i]=150
    elif i>60 and i <=75: rinp[i]=150-(i-60)*(50.0/15.0)
    elif i>95 and i <=110:rinp[i]=150
    else: rinp[i]=100.0 #fig 8.5
#------------------------------------------

for k in range(0,knum):
    #READ R
    r=rinp[k] #目標値
    #READ b
    #amplitude=0.23
    b=1 #減衰係数 緩やかに変化する　fig.8-7
    if k<=80:
        b=1-(1-np.cos(2*np.pi/80.0*k))*amplitude
    if k>80:
        b=1+(1-np.cos(2*np.pi/80.0*k))*amplitude
    Plot1[k]=b #Plot
    #
    #bにより変化するパラメータ
    p=np.exp(-b*Tsample)
    b0=(p+b*Tsample-1)/(b*Tsample*(1-p)) #Eq.8-22
    b1=(1-p-p*b*Tsample)/(b*Tsample*(1-p)) #Eq.8-22
    B[0,0]=b0;B[1,0]=b1+d1*b0;B[2,0]=d1*b1 #Eq.8-30
    #F[0,0]=c0;F[1,0]=c1;F[2,0]=d1+1+p;F[3,0]=-p
    #F0[0,0]=c1;F0[1,0]=d1+1+p;F0[2,0]=-p #Eq.8-32

    #READ V #外乱
    if k>80 and k<=90:
        V=300.0;V=200;V=50 #step状の外乱
    else: V=0 
    Plot2[k]=V #Plot
    #   
    #時変プラントの記述
    P0[0,0]=1.0;P0[0,1]=(1-p)/b**2;P0[1,0]=0.0;P0[1,1]=p #状態変数
    Q0[0,0]=(p+b*Tsample-1)/b**2;Q0[1,0]=(1-p) #状態変数
        #
    # R=(𝑟(𝑘),𝑟(𝑘−1),𝑟(𝑘−2) )
    # B=(𝑏0,(𝑏1+𝑑1*𝑏0),𝑑1*𝑏1)
    # W=(𝑢(𝑘),𝑢(𝑘−1),𝑦(𝑘),𝑦(𝑘−1))
    #
    R[2,0]=R[1,0];R[1,0]=R[0,0];R[0,0]=r # 指令値    
    W[3,0]=W[2,0];W[2,0]=y0;W[1,0]=W[0,0] # 入出力履歴
    #
    # F0=(𝑐1,(𝑑1+1+𝑝),(−𝑝))  F=(𝑐0,𝑐1,(𝑑1+1+𝑝),(−𝑝))
    # W0=(𝑢(𝑘−1),𝑦(𝑘),𝑦(𝑘−1))  W=(𝑢(𝑘),𝑢(𝑘−1),𝑦(𝑘),𝑦(𝑘−1))
    #
    F0 = F[1:4];W0=W[1:4] #copy 1行～3行
    U=(-np.dot(F0.T,W0)+np.dot(B.T,R))/F[0,0] #Eq.8-35 (F[0,0]=c0_hat)
    #U=(-np.dot(F0.T,W0)+np.dot(B.T,R))/0.01 #Eq.8-35 (F[0,0]=c0_hat)
    u[k]=U
    W[0,0]=U[0,0]
    #
    X=np.dot(P0,X)+np.dot(Q0,(U+V))
    y0=X[0,0]; y[k]=y0
    #
    #e=((y0+d1*W[3,0])-np.dot(F.T,W))/(1+G*np.dot(W.T,W)) #Eq.8-41(書籍)
    e=((y0+d1*W[2,0])-np.dot(F.T,W))/(1+G*np.dot(W.T,W)) #Eq.8-41
    #
    F=F+G*e *W #Eq.8-36
    #
    # List 8-2 END

    #################################################
    #######     　　有限整定応答の計算        ########
    ################################################# 
    # p104.pyのLISTを利用

    #input
    Xoff[0,0]=rinp[k]; Xoff[1,0]=0.0 #
    #           
    U1[0]=-np.dot(Gk,(X1-Xoff)) # Gk　有限整定制御用Gain
    #
    X1=np.dot(P0,X1)+np.dot(Q0,(U1+V)) #P0,Q0 <--- 時変プラント
    y1[k]=np.dot(C,X1) #Plot用
    #
    # End K Loop

###########################################################
#　　　　　　　　　　　    PLOT    　　　　　　　　　　　   #
############################################################
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
ax1.plot(t,y,'-+r',label="y(k)")  
ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)")
ax1.plot(t,y1,'c',linestyle='dashed',label="y1(k)")
ax1.plot(t,Plot1*40,'k',linestyle='dashed',label="b(k)") 
ax1.plot(t,Plot2,drawstyle='steps-post',color='y', linestyle='dashed', marker='',label="V(k)")  
ax1.plot(t,Plot3*40,'k',linestyle='dotted')  #補助線
strg0="d1= {:.5g}, Gain= {:.5g},".format(d1,G)
plt.title("図8-7 模型追跡適応サーボ系(MRAC & 有限整定): "+strg0, fontname="MS Gothic")

#Ymax=np.amax(y); Ymin=0.0
#strg1=" Gain: , {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}".format(G[0,0],G[0,1],G[0,2],G[0,3],G[0,4],G[0,5],G[0,6])
#strg2=" FB  : {:.5g}, {:.5g}, {:.5g},..p(^(n-1))......".format(F[0,0],F[1,0],F[2,0])

#Ymax=1.6; Ymin=-0.1
#xp=knum*2/10; yp=Ymax*4/10  #plt.textの位置座標
#plt.text(xp,yp, strg1 ) #
#plt.text(xp,yp-Ymax*1/10, strg2, fontname="MS Gothic" ) 

#plt.text(xp,yp, strg1 ) #
#
#
#plt.xlim(0,knum)
plt.ylim(0,180)
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
#ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="d(k)")
plt.ylabel("input ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
# 表示
plt.show()  