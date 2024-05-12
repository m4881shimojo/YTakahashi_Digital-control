#digital control 
#デジタル制御　高橋安人
#20240425 shimojo
#p159　図8-5 有限整定適応制御
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#--------------------------------------------------------#
#                  Begin Main                            #
#--------------------------------------------------------#
#
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=4,  floatmode="fixed")#　=True 指数表記禁止
#
Dnum=4 # g1,g2,g3,g4
n=4 #状態変数　(積分器を加えると+mとなる）
m=1 # m個のu入力 (今回意味はない)
knum=120 #計算の回数上限

P=np.zeros((n,n));Q=np.zeros((n,m));C=np.zeros((m,n))
#
rinp=np.zeros(knum)
X=np.zeros((n,1)) #状態変数 for plant
#X0=np.zeros((n,1)) #状態変数 for model

u=np.zeros((knum,m)) #入力
#d=np.zeros((knum,m)) #刻み入力
U=np.zeros((m,1))
V=np.zeros((m,1)) #noise
y=np.zeros((knum,1)) #出力 for plant　dy
#y0=np.zeros((knum,1)) #出力 for model dy0
#Py=np.zeros((knum,1)) #出力 for plant： PLOT用
#Py0=np.zeros((knum,1)) #出力 for model： PLOT用

#G=np.zeros((m,n)) #Riccati-gain
Gk=np.zeros((m,n)) #有限整定gain　<----g
Gk0=np.zeros((m,n)) #有限整定gain <----g0からの推定値

#from p157.py
#MRAS 関係配列
g=np.zeros(Dnum);g0=np.zeros(Dnum)
Um=np.zeros(Dnum) #p153でのu
Kg=100.0 #p153でのKj

Pum=np.zeros(knum) # for Plot
Pg0=np.zeros((knum,Dnum)) #g0の推定値 Plot用
Padj=np.zeros(knum) # MRAS 実行　Flag
#
# plant parameter given
p_hat=0.6
g1=0.1;g2=0.3;g3=0.48;g4=0.4
g[0]=g1;g[1]=g2;g[2]=g3;g[3]=g4

#model parameter Estimate
p0_hat=0.0
g01=0.5;g02=0.4;g03=0.3;g04=0.15
g0[0]=g01;g0[1]=g02;g0[2]=g03;g0[3]=g04
#
# plant 
a1=-p_hat
b1=g1;b2=g2-p_hat*g1;b3=g3-p_hat*g2;b4=g4-p_hat*g3

P=np.array([[0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [0,0,0,-a1]]) 
Q=np.array([[g[0]],
            [g[1]],
            [g[2]],
            [g[3]]]) 
C=np.array([[1,0,0,0]]) 

#model
p0_hat=0.0 #initial value
a01=-p0_hat
b01=g01;b02=g02-p0_hat*g01;b03=g03-p0_hat*g02;b04=g04-p0_hat*g03


print("---------------data----------------")
print("p_hat=",p_hat)
print("g= ",g)
print("p0_hat=",p0_hat)
print("g= ",g0)
print("-----------------------------------")
############################################
#　　　　　　　dc-Gainを求める              #
############################################
#calculate dcGain
#p_hat fig8-2 で与えられた値
#p1=(G(1)-(g1+g2+....gn))/(G(1)-(g1+g2+....g(n-1)) eq.4-16
#G(1)=g1+g2+....gn-1+gn/(1-p) #eq.4-15
gsum=0.0
for k in range(0,Dnum-1):
    gsum+=g[k]
dcGain=gsum+g[Dnum-1]/(1-p_hat)
print(" dcGain=" ,dcGain)
print(" b4,b3,b2,b1    ----->",b4,b3,b2,b1)
print(" b04,b03,b02,b01----->",b04,b03,b02,b01)


#################################################
######　  （リカチ行列の収束計算は不要　）   ######
######　    有限整定制御のため　決め打ち     ######
#################################################
#参考：
#Riccati-Gain=[[1.39639 0.00000 1.39639 1.20140 0.65164]]
#有限整定Gain=  [[ 1.32979  0.00000  1.32979  1.32979 -0.10213]]
#
#plant <-- parameter given
a1=-p_hat
b1=g1;b2=g2-p_hat*g1;b3=g3-p_hat*g2;b4=g4-p_hat*g3
#model <-- parameter estimate
p0_hat=0.0 # initila value
a01=-p0_hat
b01=g01;b02=g02-p0_hat*g01;b03=g03-p0_hat*g02;b04=g04-p0_hat*g03
#
# 式7-29から求めた値
# plant
K0=1/(b1+b2+b3+b4)
K4=1+p_hat-K0*(g1+g2+g3)
print(" Plant: K0,K4= ",K0,K4)
#model
K00=1/(b01+b02+b03+b04)
K04=1+p0_hat-K00*(g01+g02+g03)
#K04=0.111
print(" Model: K00,K04= ",K00,K04)
#
#Initial FTSC-Gains
#書籍
K00=0.741;K04=.111 # ok
#
Gk[0,0]=0.0;Gk[0,1]=K0;Gk[0,2]=K0;Gk[0,3]=K4
#
K0g=K00 #応答計算でつかう
Gk0[0,0]=0.0;Gk0[0,1]=K00;Gk0[0,2]=K00;Gk0[0,3]=K04

print("有限整定Gain= ",Gk)
print("推定値Gain0= ",Gk0)
print("#------------------------------------#")

#################################################
######　    初期　応答の計算   　  　　　　　######
######      Gain G[K0,k1,k2] 計算済み       ######
#################################################
# 
#--------------　ステップ入力--------------
for i in range(0,knum):
    if i>25 and i<75 :rinp[i]=150.0
    elif i>=75 :rinp[i]=100.0
    else: rinp[i]=50.0 #fig 8.5
#------------------------------------------

#応答の計算
Noise_start=50
x0=np.zeros((knum,m)) #小文字のx0　
K00sum=0.0;K04sum=0.0 #K Gain 絶対値和平均
nadj=0.0 #gain変更回数

for k in range(0,knum): #
    #
    y[k]=np.dot(C,X)  #plant
    #
    #入力u[k]の算出    
    if k==0: x0[k]=(rinp[k]-y[k])
    else:x0[k]=x0[k-1]+(rinp[k]-y[k])
    #
    u[k]=K0g*x0[k]-np.dot(Gk0,X) #状態FB
    U[0]=u[k] #計算のため置き換え

    #外乱 Step   
    if k>=50 and k<=100 :V[0]=-50. 
    else: V[0]=0.0
    
    X=np.dot(P,X)+np.dot(Q,(U+V)) #plant
    y[k]=np.dot(C,X)  #plant
    if k!=knum-1 : x0[k+1]=y[k]
    #

#----------------------------------------------------------#    
#         --------------------MRAS---------------------    #
#----------------------------------------------------------# 
    if (k % 3)==0 :
        #
        #print("回数k=",k,"p0_hat=",p0_hat)#
        Pg0[k,:]=g0 # Plot
        Pum[k]=Um[0]  # Plot
        #
        #Begin List8-1
        #入力Uをどうとるか？
        #gとg0の値を入力を与え、それによって生ずる誤差により漸近させる
        #以下の方法では、ランダム信号を与える方が良い　

        for j in range(Dnum-1,0,-1):            
            Um[j]=Um[j-1]    
        
        Um[0]=u[k,0] #+ 1.0*(np.random.rand()-0.5)
        #Um[0]=np.random.rand()-0.5 #、0~1の範囲(一様分布)1個の数値
        #
        ym=0.0;ym0=0.0;Dd=1.0# Dd-->K*U[j]**2 Eq.8-14
        #        
        for j in range(0,Dnum):
            ym=ym+g[j]*Um[j]
            ym0=ym0+g0[j]*Um[j]
            Dd=Dd+Kg*Um[j]**2
        e=(ym-ym0)/Dd
        #
        for j in range(0,Dnum):
            g0[j]=g0[j]+Kg*e*Um[j] #eq.8-14        #
        #
        # End  List8-1 

        #-------------------------------------------#
        #calculate p0_hat#
        gsum=0.0;gsum0=0.0
        for j in range(0,Dnum-1):
            gsum0=gsum0+g0[j]
        p0_hat=(dcGain-gsum0-g0[Dnum-1])/(dcGain-gsum0)

        #p0_hat
        a01=-p0_hat
        #b01=g01;b02=g02-p0_hat*g01;b03=g03-p0_hat*g02;b04=g04-p0_hat*g03
        b01=g0[0];b02=g0[1]-p0_hat*g0[0];b03=g0[2]-p0_hat*g0[1];b04=g0[3]-p0_hat*g0[2]
        
        #calculate K gain        
        K00=1/(b01+b02+b03+b04)
        K04=1+p0_hat-K00*(g0[0]+g0[1]+g0[2])
        #-------------------------------------------#
        #print(" Model: K00,K04= ",k,K00,K04)
        
        # Gain change
        K00change=abs((Gk0[0,1]-K00)/Gk0[0,1])
        K04change=abs((Gk0[0,3]-K04)/Gk0[0,3])

        if k >=20 and k <30 :
            nadj=nadj+1.0 #adjust 適応回数            
            K00sum=K00sum+K00change;K04sum=K04sum+K04change
               
        #print("-----------------k=",k,K00change,K04change)#Debug    
        #
        if k <30 :
            K0g=K00
            Gk0[0,0]=0.0;Gk0[0,1]=K00;Gk0[0,2]=K00;Gk0[0,3]=K04
            Padj[k]=10.0 #Plot用　適応表示するのみ

        #Gain変動が平均絶対和を超える？                     
        #変化率が平均より”Trigerth1”大きいか､"Trigerth2"より小さい--->制御Gainを書き換え
        Trigerth1=0.1;Trigerth2=3.0 #分かり易くするため、ここに置く

        if k>=30 :
            Triger1=(K00change > Trigerth1*K00sum/nadj and K00change< Trigerth2*K00sum/nadj) 
            Triger2=(K04change > Trigerth1*K04sum/nadj and K04change< Trigerth2*K04sum/nadj)
            #print("T---->",Triger1,Triger2) #Debug
        
        if k>=30 and (Triger1 or Triger2):
            K0g=K00
            Gk0[0,0]=0.0;Gk0[0,1]=K00;Gk0[0,2]=K00;Gk0[0,3]=K04
            Padj[k]=20.

    # End k_loop

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
ax1.plot(t,rinp,drawstyle='steps-post',color='b', linestyle='dashed', marker='',label="r(k)")
ax1.plot(t,Padj,'.k',label="Adjust")

strg0="30Step以降でFB_Gain変化が範囲内のとき適応動作"
plt.title("図8-5 有限整定適用制御:"+strg0, fontname="MS Gothic")

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
#plt.ylim(-0.05,1.2)
plt.ylabel("Response ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い
#
####222222222222222222222########
# ax2　PLOT
####222222222222222222222########
ax2 = plt.subplot(ss2)
#ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,u,drawstyle='steps-post',color='g', linestyle='dashed', marker='.',label="u(k)")
#ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="d(k)")
plt.ylabel("Response ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='upper right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()

############################################################
#                figure 2                                  #
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
ax1.plot(t,Pg0[:,0],'*c',label="g1(k)") 
ax1.plot(t,Pg0[:,1],'*r',label="g2(k)") 
ax1.plot(t,Pg0[:,2],'*b',label="g3(k)") 
ax1.plot(t,Pg0[:,3],'*y',label="g4(k)") 

strg0="Kg={:.3g}".format(Kg)
plt.title("図8-5 有限整定適用制御(MRASによるプロセス同定):"+strg0, fontname="MS Gothic")

#Ymax=np.amax(Pg0); Ymin=0.0
#Ymax=0.6; Ymin=0
#xp=knum*3/10; yp=Ymax*4/10  #plt.textの位置座標
#strg1=" g[k]: g1={:.5g}, g2={:.5g}, g3={:.5g}, g4={:.5g}".format(g[0],g[1],g[2],g[3],)
#plt.text(xp,yp, strg1 ) #
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
ax2.plot(t,Pum,drawstyle='steps-post',color='g', linestyle='dashed', marker='.',label="u(k)")
#ax2.plot(t,d,drawstyle='steps-post',color='b', linestyle='dashed', marker='*',label="d(k)")
plt.ylabel("input ")
plt.xlabel("step (k)")

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い

plt.tight_layout()

# 表示
plt.show()    