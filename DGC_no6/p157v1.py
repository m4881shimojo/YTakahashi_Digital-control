#digital control 
#デジタル制御　高橋安人
#20240414shimojo

#157 Fig8-3
#MARSによるプロセス同定
#8.3　MARSによる同定と適応制御への応用
#p155 list8-1
#
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

knum=60;Dnum=4
K=100.0
g=np.zeros(Dnum);g0=np.zeros(Dnum)
U=np.zeros(Dnum)
umras=np.zeros(knum) #input for MRAS
yg0=np.zeros((knum,Dnum)) #g0の推定値 Plot用
np.set_printoptions(precision=4,  floatmode="fixed")#　=True 指数表記禁止
#
#fig8-3 data
#Plant
pHat=0.6 #p for Plant
g=np.array([0.1,0.3,0.48,0.4])
#Model
p0Hat=0 #p for Model
g0=np.array([0.5,0.4,0.3,0.15]) #Model
##########################################################
#                    Normalize                           #
#  fig8-3 dataのNormalize　--->step応答が１に収束する     #
#   すなわち、dcGainが１となるようにNormalizeする         #
##########################################################
############################################
#　　　　　　　dc-Gainを求める              #
############################################
#calculate dcGain
#pHat fig8-2 で与えられた値
#p1=(G(1)-(g1+g2+....gn))/(G(1)-(g1+g2+....g(n-1)) eq.4-16
#G(1)=g1+g2+....gn-1+gn/(1-p) #eq.4-15
gsum=0.0
for k in range(0,Dnum-1):
    gsum+=g[k]
dcGain=gsum+g[Dnum-1]/(1-pHat)
print("dcGain=" ,dcGain)

#Normarized Plant Gain
for k in range(0,Dnum):
    g[k]=g[k]*1.0/dcGain #1.0-->10.0 すると
    g0[k]=g0[k]*1.0/dcGain #1.0-->10.0

print("g= ",g)
print("g0= ",g0)

#ここで推定しても意味ない（あくまで参考値）
gsum=0.0;gsum0=0.0
for k in range(0,Dnum-1):
    gsum=gsum+g[k];gsum0=gsum0+g0[k]
p_hat=(dcGain-gsum-g[Dnum-1])/(dcGain-gsum) #eq.4-16による推定値
p0_hat=(dcGain-gsum0-g0[Dnum-1])/(dcGain-gsum0)
print("p_hat= ",p_hat,"p0Hat= ",p0_hat)
#End　Normalize

##########################################################
#                         MRAS                           #
##########################################################
#Begin List8-1
for i in range(0,knum):
    yg0[i,:]=g0
    
    for j in range(Dnum-1,0,-1):
        #print(j)
        U[j]=U[j-1]    
    U[0]=np.random.rand()-0.5 #、0~1の範囲(一様分布)1個の数値
    umras[i]=U[0] #for PLOT MRASを行うための乱数入力
    y=0;y0=0;Dd=1# Dd-->K*U[j]**2 Eq.8-14
    #print(U[1])
    
    for j in range(0,Dnum):
        y=y+g[j]*U[j]
        y0=y0+g0[j]*U[j]
        Dd=Dd+K*U[j]**2
    e=(y-y0)/Dd
    for j in range(0,Dnum):
        g0[j]=g0[j]+K*e*U[j] #eq.8-14
    #
    #yg0[i,:]=g0
# End  List8-1
print("g0=",g0)

#calculate dcGain and pm_hat
#仮定：normalize したので　dcGain=1
#p1=(G(1)-(g1+g2+....gn))/(G(1)-(g1+g2+....g(n-1))
gsum=0.0;gsum0=0.0
for k in range(0,Dnum-1):
    gsum=gsum+g[k];gsum0=gsum0+g0[k]
p_hat=(1-gsum-g[Dnum-1])/(1-gsum)#eq.4-16による推定値
p0_hat=(1-gsum0-g0[Dnum-1])/(1-gsum0)
print("\n ------------pの推定値--------------")
print("p_hat= ",p_hat,"p0_hat= ",p0_hat)


############################################
#　　　　　4.4 時系列からの状態空間形　　　  #
#             page 72                      #
############################################
 
P=np.array([[0.,1.,0.,0.],
            [0.,0.,1.,0.],
            [0.,0.,0.,1.],
            [0.,0.,0.,pHat]])
q=np.array([[g[0]], #p72に説明
            [g[1]],
            [g[2]],
            [g[3]]])
c=np.array([1.,0.,0.,0.]) #
#model
#p0_hat
#p0Hat=0
P0=np.array([[0.,1.,0.,0.],
            [0.,0.,1.,0.],
            [0.,0.,0.,1.],
            [0.,0.,0.,p0_hat]])
q0=np.array([[g0[0]], #p72に説明
            [g0[1]],
            [g0[2]],
            [g0[3]]])
c0=np.array([1.,0.,0.,0.]) #

n=Dnum
u=np.ones(knum) #step input
Xk1=np.zeros((n,1))
Xk=np.zeros((n,1))
y=np.zeros((knum,2))# plotのデータアレイ
Y=np.zeros((1,1))
  #応答の算定　（X(k+1)=PX(k)+qu(K)）
#plant
for k in range(0,knum):
    Xk1=np.dot(P,Xk)+u[k]*q   
    Y=np.dot(c,Xk1)
    y[k,0]=Y[0]
    Xk=np.copy(Xk1)    
#end K loop
Xk1=np.zeros((n,1))
Xk=np.zeros((n,1))
#model
for k in range(0,knum):
    Xk1=np.dot(P0,Xk)+u[k]*q0   
    Y=np.dot(c0,Xk1)
    y[k,1]=Y[0]
    Xk=np.copy(Xk1)    
#end K loop


#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
############################################################
#                figure 1                                  #
############################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6,6)) # Figureの初期化
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
ax1.plot(t,yg0[:,0],'-*c',label="g1(k)") 
ax1.plot(t,yg0[:,1],'-*r',label="g2(k)") 
ax1.plot(t,yg0[:,2],'-*b',label="g3(k)") 
ax1.plot(t,yg0[:,3],'-*m',label="g4(k)") 
#ax1.plot(t,y[:,0],'--r',label="plant(k)")  #input
#ax1.plot(t,y[:,1],'--b',label="model(k)")  #input
#ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)") 

strg0="MRAS Gain K={:.3g}".format(K)
plt.title("図8-3 MRASによるプロセス同定 :"+strg0, fontname="MS Gothic")

#Ymax=np.amax(yg0); Ymin=0.0
#Ymax=0.6; Ymin=0
#xp=knum*3/10; yp=Ymax*4/10  #plt.textの位置座標
#strg1=" g[k]: g1={:.5g}, g2={:.5g}, g3={:.5g}, g4={:.5g}".format(g[0],g[1],g[2],g[3],)
#plt.text(xp,yp, strg1 ) #
#
#
#plt.ylim(Ymin,Ymax)
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
ax2.plot(t,umras,drawstyle='steps-post',color='g', linestyle='dashed', marker='.',label="u(k)")
#ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="d(k)")
plt.ylabel("input ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()
############################################################
#                figure 2                                  #
############################################################
fig = plt.figure(figsize=(5.5,5.5)) # Figureの初期化
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
ax1.plot(t,y[:,0],'-or',label="plant(k)")  #step response for plant
ax1.plot(t,y[:,1],'--b',label="model(k)")  #step response for model
#ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)") 

strg0="p_hat={:.5g}, p0_hat={:.5g}".format(p_hat,p0_hat)
plt.title("図8-3 プロセス同定後のStep応答 :"+strg0, fontname="MS Gothic")

#Ymax=np.amax(yg0); Ymin=0.0
#Ymax=0.6; Ymin=0
#xp=knum*3/10; yp=Ymax*4/10  #plt.textの位置座標
#strg1=" g[k]: g1={:.5g}, g2={:.5g}, g3={:.5g}, g4={:.5g}".format(g[0],g[1],g[2],g[3],)
#plt.text(xp,yp, strg1 ) #
#
#
#plt.ylim(Ymin,Ymax)
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
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()



# 表示
plt.show()    