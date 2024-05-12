#digital control 
#デジタル制御　高橋安人
#20240430 shimojo
#p174 Fig 8-8
#同定および適応制御
#8.5　最小2乗同定アルゴリズム
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=4 #3次系
nump=4 #W,F
m=1 # m個のu入力
knum=10 #サンプル数


F=np.zeros((n,m));F0=np.zeros((n,m)) #縦V
H=np.zeros((n,n))
Q=np.zeros((n,n)) #単位行列
G=np.zeros((n,m)) #Gain
#
#X=[y(j-1),y(j-2),u(j-1),u(j-2)]
X=np.zeros((n,1))
#Y=np.zeros((1,1));Y0=np.zeros((1,1) #vectorとなる
Plot=np.zeros((knum,n)) # for Plot
yPlot=np.zeros((knum,3))

#Begin List 8-3
#INPUT F
F[0,0]=-1.6;F[1,0]=0.8;F[2,0]=0.4;F[3,0]=0.6 #Plant
F[0,0]=1.6;F[1,0]=-0.8;F[2,0]=0.4;F[3,0]=0.6 #Plant
#F=[-a1,-a2,b1,b2]である　Eq.8-43
#上記Fの値は、表8-5の値；a(1)=-1.6;a(2)=0.8;b(1)=0.4;b(2)=0.6
#a(1)=-a1;a(2)=-a2;b(1)=b1;b(2)=b2 のことか？？
#a(1)=a1;a(2)=a2;b(1)=b1;b(2)=b2 のことだろう
#少々記述がまぎらわしい

#INPUT c,w
c=200;w=1 #CASE 1
#c=1000;w=0.8 #CASE 2
#
#X=[y(j-1),y(j-2),u(j-1),u(j-2)]
X=np.zeros((n,1));y=0;U=0.2
np.random.seed(314-3) #seedは整数。常に同じ乱数時系列を発生させるため？

for k in range(0,2): #書籍では(0,2)-->(0,4)
    #X=[y(j-1),y(j-2),u(j-1),u(j-2)]
    X[3,0]=X[2,0]; X[2,0]=np.random.rand();X[1,0]=X[0,0];X[0,0]=y
    Y=np.dot(F.T,X) #-a1*y(j-1)-a2*y(j-2)+b1*u(j-1)+b2*u(j-2) #Eq.8-43
    y=Y.item() #Y-->(1,1) scalar_value変換　shimojo
print(X);print("y=",y);print("F=",F)

F0=np.zeros((n,m));Q=np.eye(n)#F0-->F_hat
H=c*Q #Eq.8-53

for k in range(0,knum):
    D=1+np.dot(X.T,np.dot(H,X))*(1/w) #
    G=(1/(w*D))*np.dot(H,X) #Eq.8-52

    Y=np.dot(X.T,F);Y0=np.dot(X.T,F0) #plan & model output   
    y=Y.item();y0=Y0.item()#scalar_value変換　shimojo
    if k !=knum-1:yPlot[k+1,0]=y; yPlot[k+1,1]=y0 #Plot用に保存
   

    F0=F0+(y-y0)*G # Eq.8-52 係数ベクトル式の最適推定-->F0 
    if k !=knum-1: Plot[k+1,:]=-F0.T #Plot用に保存

    H=(1/w)*np.dot((Q-np.dot(G,X.T)),H) #Eq.8-51 H(k+1)へ更新
    X[3,0]=X[2,0]; X[2,0]=np.random.rand();X[1,0]=X[0,0];X[0,0]=y
    if k !=knum-1:yPlot[k+1,2]=X[2,0]#u[k]
    #
    E = np.sum(np.abs(F-F0)) # F & F_hatの誤差
    #print(E)
#End List 8-3
print("k step=",knum)
print("F0=",F0)


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
ax1.plot(t,Plot[:,0],'-+r',label="a1")  
ax1.plot(t,Plot[:,1],'-+b',label="a2")
ax1.plot(t,Plot[:,2],color='c', linestyle='dashed', marker='.',label="b1")
ax1.plot(t,Plot[:,3],color='k', linestyle='dashed', marker='.',label="b2") 

strg0="c= {:.5g}, w= {:.5g},".format(c,w)
plt.title("図8-8 最小2乗反復同定の例 "+strg0, fontname="MS Gothic")

#plt.ylim(-1,2)
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
ax2.plot(t,yPlot[:,0],'-+r',label="y(k)")
ax2.plot(t,yPlot[:,1],drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="y0(k)")
ax2.plot(t,yPlot[:,2],drawstyle='steps-post',color='m', linestyle='dashed', marker='.',label="u(k)")
plt.ylabel("Response ")
plt.xlabel("step (k)")
plt.title("反復同定の計算過程での y(k),y0(k) & u(k) ", fontname="MS Gothic")
#plt.ylim(0,50)

plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
# 表示
plt.show()  
