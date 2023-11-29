#digital control 
#デジタル制御　高橋安人
#
#p69図4-4をPLOTする　20231106
#伝達関数：理論式からと実験データからでの比較　20231124
#(4-10)G(s)の応答を算出する
#T=4の場合のPLOT
#L=3のため、T=4,L1=3とする。
#L=N*T+L1
#
#(4-11)G(z)に変換する。その時#(4-12)のパラメータ計算
#
#応答計算はp35を用いる
#図3-2ステップ応答に及ぼすゼロの影響　page.35
# shimojo 20231015

#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#from numpy.linalg import inv
#np.set_printoptions(formatter={'float': '{:.2f}'.format})
#
knum=30
Y=np.zeros(knum)
U=np.ones(knum)
g=np.zeros(knum) #gj=yj-y(j-1)
#
np.set_printoptions(precision=3, suppress=True)
#(4-12),Fig 4-4から選んだパラメータ
T1=10;T2=6
T=4;L1=3;N=0 #表4-5ではL=3-->N=0,L1=3
L=N*T+L1
#L=3
#K=1としてる
#G(z)=K(b1z^2+b2z+b3)/{z^(N+1)(z^2+s1z+a2)}
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常

print("\nN,T,L,L1 :",N,T,L,L1)
#係数を(4-12)を用いて決める
#
p1=np.exp(-T/T1);p2=np.exp(-T/T2)
a1=-(p1+p2);a2=p1*p2
d1=np.exp(L1/T1);d2=np.exp(L1/T2);r=T2/T1#r NOT EQ "0"
b1=1-(p1*d1-r*p2*d2)/(1-r)
b2=(p1*d1*(1+p2)-r*p2*d2*(1+p1))/(1-r)-(p1+p2)
b3=p1*p2*(1-(d1-r*d2)/(1-r))

#a1= -1.535262063651771 a2= 0.5866462195100318 # 
#a1=-1.535; a2=0.587#書籍
#b1= 0.00762904224602201 ;b2= 0.03840881647631877;b3=0.005346297135919452 # 
#b1=0.008;b2=0.038;b3=0.005 #書籍
#
print("\na1,a2 :",a1,a2) #;a_para=str(a1)+","+str(a2)
print("\nb1,b2,b3 :",b1,b2,b3) #;b_para=str(b1)+","+str(b2)+","+str(b3)
#

#########################################################
#######     ここから応答の計算 理論値から伝達関数   #######
#######      p68工業用プロセス応答のよくある例      #######
#########################################################
#
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常
#z(^-1)が掛かっていると考えると
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-2]+b2*U[k-3]+b3*U[k-4]) 
Y[0]=0.0
Y[1]=-a1*Y[0]
Y[2]=-(a1*Y[1]+a2*Y[0])+(b1*U[0])  
Y[3]=-(a1*Y[2]+a2*Y[1])+(b1*U[1]+b2*U[0])  
#Y[4]=-(a1*Y[3]+a2*Y[2])+(b1*U[2]+b2*U[1]+b3*U[0]) 

for k in range(4,knum): 
    Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-2]+b2*U[k-3]+b3*U[k-4])  
    #print(k,Y[k])
#
g[0]=Y[0]
for j in range(1,knum):
    g[j]=(Y[j]-Y[j-1])
    #if j>1: print(j, g[j]/g[j-1])
print("\ng=",g)

#########################################################
#######     ここから応答の計算 実験値から伝達関数   #######
#######     p68工業用プロセス応答                  #######
#########################################################
#
u=np.ones(knum)
y=np.zeros(knum)
YY=np.zeros(knum)# plotのデータアレイ 
gg=np.zeros(knum)# plotのデータアレイ 
#
#page 70 表4－5 書籍データから
n=6
p=0.708
g1=0.008;g2=0.128;g3=0.183;g4=0.172;g5=0.141;g6=0.108
#以上　書籍データから
#以下理論式から計算した値
#acalculate g()　p70&p35forT2Final.py
#g= [0.       0.       0.007629 0.127942 0.1827   0.172238 0.141007 0.107639
# 0.078888 0.056339 0.03954  0.027416 0.018846 0.012873 0.008752 0.00593
# 0.004008 0.002703 0.001821 0.001225 0.000823 0.000553 0.000371 0.000249
# 0.000167 0.000112 0.000075 0.00005  0.000034 0.000023]
#
# 理論式からのg1,g2,g3.....
#g1=0.007629;g2=0.127942;g3=0.1827;g4= 0.172238;g5= 0.141007;g6= 0.107639
#
#応答　(4-14)式より
#Y[k]=p*Y[k-1]+g1*U[k-1]+g2*U[k-2]+g3*U[k-3]+g4*U[k-4]+g5*U[k-5]+g6*U[k-6]-p*(g1*U[k-2]+g2*U[k-3]+g3*U[k-4]+g4*U[k-5]+g5*U[k-6])
#step input
U=np.ones(knum)
#
y[0]=0.0
y[1]=p*y[0]+g1*U[0]
y[2]=p*y[1]+g1*U[1]+g2*U[0]-p*(g1*U[0])
y[3]=p*y[2]+g1*U[2]+g2*U[1]+g3*U[0]-p*(g1*U[1]+g2*U[0])
y[4]=p*y[3]+g1*U[3]+g2*U[2]+g3*U[1]+g4*U[0]-p*(g1*U[2]+g2*U[1]+g3*U[0])
y[5]=p*y[4]+g1*U[4]+g2*U[3]+g3*U[2]+g4*U[1]+g5*U[0]-p*(g1*U[3]+g2*U[2]+g3*U[1]+g4*U[0])
#Y[k]=p*Y[k-1]+g1*U[k-1]+g2*U[k-2]+g3*U[k-3]+g4*U[k-4]+g5*U[k-5]+g6*U[k-6]-p*(g1*U[k-2]+g2*U[k-3]+g3*U[k-4]+g4*U[k-5]+g5*U[k-6])

for k in range(6,knum):
    y[k]=p*y[k-1]+g1*U[k-1]+g2*U[k-2]+g3*U[k-3]+g4*U[k-4]+g5*U[k-5]+g6*U[k-6]-p*(g1*U[k-2]+g2*U[k-3]+g3*U[k-4]+g4*U[k-5]+g5*U[k-6])

    
for j in range(0,knum):
    if j==0: gg[j]=0.0
    else: gg[j]=y[j]-y[j-1]

#########################################################
#　PLOT 　グラフを描く                                   #
#########################################################
#　グラフを描く
fig = plt.figure(figsize=(8,6)) #横　&　縦
ax1 = plt.subplot(111)
t=np.arange(0,knum)
ax1.plot(t,Y,'-or')
ax1.plot(t,y,'-ok')  
ax1.plot(t,g*5.0,'-*b')
ax1.plot(t,gg*5.0,'-*c')

strg1="g1={:.3g},g2={:.3g},g3={:.3g},g4={:.3g},g5={:.3g},g6={:.3g},p={:.3g}".format(g1,g2,g3,g4,g5,g6,p)
strg2="T={:.3g}, T1={:.3g}, T2={:.3g}, L={:.3g}, L1={:.3g}".format(T,T1,T2,L,L1)
plt.title("図4-4 プロセス応答"+strg2, fontname="MS Gothic")
Ymax=1.2; Ymin=-0.3
plt.ylim(Ymin,Ymax)
plt.xlim(0,knum)
plt.ylabel("Responce y(k)")
plt.xlabel("Step")
xp=knum*1/10; yp=Ymin+Ymax*2/10  #plt.textの位置座標
plt.text(xp,yp, "a1,a2={:.3g},{:.3g}".format(a1,a2) )
plt.text(xp,yp-Ymax/15, "b1,b2,b3={:.3g},{:.3g},{:.3g}".format(b1,b2,b3))
#xp=knum*3/10; yp=Ymax*6/10  #plt.textの位置座標
plt.text(xp,yp-Ymax*2/15, strg1 )

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