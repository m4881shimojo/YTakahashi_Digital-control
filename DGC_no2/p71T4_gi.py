#digital control 
#デジタル制御　高橋安人
#p69図4-4をPLOTする　20231106
#(4-10)G(s)の応答を算出する
#T=4の場合のPLOT
#L=3のため、N=0,T=4,L1=3とする。
#L=N*T+L1
#
#(4-11)G(z)に変換する。その時#(4-12)のパラメータ計算
#
#応答計算はp35を用いる
# shimojo 20240203

#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#from numpy.linalg import inv
#np.set_printoptions(formatter={'float': '{:.2f}'.format})
np.set_printoptions(precision=5, suppress=True)
#np.set_printoptions(precision=3,  floatmode="fixed")
#(4-12),Fig 4-4から選んだパラメータ
T1=10;T2=6
T=4;L1=3;N=0 #表4-5ではL=3-->N=0,L1=3
L=N*T+L1
#L=3
#K=1としてる
#G(z)=K(b1z^2+b2z+b3)/{z^(N+1)(z^2+s1z+a2)}
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--N=0
print("\nN,T,L,L1 :",N,T,L,L1)
#係数を(4-12)を用いて決める
#
p1=np.exp(-T/T1);p2=np.exp(-T/T2)
a1=-(p1+p2);a2=p1*p2
d1=np.exp(L1/T1);d2=np.exp(L1/T2);r=T2/T1#r NOT EQ "0"
b1=1-(p1*d1-r*p2*d2)/(1-r)
b2=(p1*d1*(1+p2)-r*p2*d2*(1+p1))/(1-r)-(p1+p2)
b3=p1*p2*(1-(d1-r*d2)/(1-r))
print("p1={:.5g}, p2={:.5g}".format(p1,p2))
#
knum=30
Y=np.zeros(knum)
U=np.ones(knum)
g=np.zeros(knum) #gj=yj-y(j-1)


#a1= -1.535262063651771 a2= 0.5866462195100318 # 
#a1=-1.535; a2=0.587#書籍

#b1= 0.00762904224602201 ;b2= 0.03840881647631877;b3=0.005346297135919452 # 
#b1=0.008;b2=0.038;b3=0.005 #書籍
#
print("\na1={:.5g}, a2={:.5g}".format(a1,a2)) #;a_para=str(a1)+","+str(a2)
print("b1={:.5g}, b2={:.5g}, b3={:.5g}".format(b1,b2,b3)) #;b_para=str(b1)+","+str(b2)+","+str(b3)


#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) 
# 

for k in range(0,knum):   
    if k==0: Y[k]=0.
    elif k==1: Y[k]=-(a1*Y[k-1])+(b1*U[k-1]) 
    elif k==2: Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]) 
    else: Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) 
#
#calculate g(k)
for j in range(0,knum):
    if j==0: g[j]=Y[j]
    else: g[j]=(Y[j]-Y[j-1])
print("\ng=",g)

############################################
#　　　　　4.4 時系列からの伝達関数　　　　   #
#             page 68                      #
############################################
#G(z)=g1z^(-1)+g2z^(-2)+.....g(n-1)z^(n-1)+g(n)z^(n)/(1-pz^(-1))
#pを算出する
#p1=(G(1)-(g1+g2+....gn))/(G(1)-(g1+g2+....g(n-1))
#今回単位ステップの平衡値は”１”となる
## G(1)=g1+g2+....gn-1+gn/1-p
# p=(gn+1)/gn
# p2=gn+1/gn  e=|p2-p1|/p1
# e<0.05に達すればOK　その時のp1(or p2) をpとする
#
#p1を求める
#p1=(G(1)-(g1+g2+....gn))/(G(1)-(g1+g2+....g(n-1))
#G(1)=1とする←理想状態
#####test#####
gsum=0
for i in range(0,knum):
    gsum+=g[i]    
print("gsum=\n",gsum)
#####test#####

gsumN1=0
for i in range(1,knum-1):
    gsumN1+=g[i-1]
    p1=(1-gsumN1-g[i])/(1-gsumN1)
    p2=g[i+1]/g[i] #g[i]=0のときあり
    err=abs(p1-p2)/p1
    print("i=",i,p1,p2,err)
# 単位ステップ応答の実測値からのG(z)推定
#i= 6 0.7078867001593836 0.7328960766436616 0.03532963181628793
p1_hat=0.7078867 #書籍とほぼ同じとした
Dnum=6 #書籍と同じとした

bj=np.zeros(knum) #b1,b2,...(4-19)

for j in range(1,Dnum):
    if j==1: bj[j]=g[j]
    else: bj[j]=g[j]-p1_hat*g[j-1]
print(bj)

############################################
#　時系列からの伝達関数を用いた応答　　　   #
#             page 68                      #
############################################
#G(z^(-1))={b1*z^(-1)+b2*z^(-2)+..+bn*z^(-n)}/(1-p*z^(-1))
#Y[k]=p*Y[k-1]+bj1*U[k-1]+bj2*U[k-2]+bj3*U[k-3]+bj4*U[k-4]+bj5*U[k-5]+bj6*U[k-6]
bj0=0;bj1=bj[1];bj2=bj[2];bj3=bj[3];bj4=bj[4];bj5=bj[5];bj6=bj[6]

Y=np.zeros(knum)
U=np.ones(knum)

p=p1_hat
for k in range(0,knum):   
    if k==0: Y[k]=0.
    elif k==1: Y[k]=Y[k]=p*Y[k-1]+bj1*U[k-1]
    elif k==2: Y[k]=p*Y[k-1]+bj1*U[k-1]+bj2*U[k-2]
    elif k==3: Y[k]=p*Y[k-1]+bj1*U[k-1]+bj2*U[k-2]+bj3*U[k-3]
    elif k==4: Y[k]=p*Y[k-1]+bj1*U[k-1]+bj2*U[k-2]+bj3*U[k-3]+bj4*U[k-4]
    elif k==5: Y[k]=p*Y[k-1]+bj1*U[k-1]+bj2*U[k-2]+bj3*U[k-3]+bj4*U[k-4]+bj5*U[k-5]    
    else: Y[k]=p*Y[k-1]+bj1*U[k-1]+bj2*U[k-2]+bj3*U[k-3]+bj4*U[k-4]+bj5*U[k-5]+bj6*U[k-6]
#

###############################################################
#　                   グラフを描く(PLOT)                       #
###############################################################
t=np.arange(0,knum)
plt.plot(t,Y,'-*r',label="Y(k)")     #最終出力 
plt.plot(t,g*5,'-*b',label="g(j)x5") 

#plt.plot(t,Y,'-or')     #最終出力 
#plt.plot(t,g,'-*r')     #最終出力 
#plt.plot(t,bj,'-*b') 
plt.title("図4-4 時系列からの伝達関数を用いた応答, T="+str(T), fontname="MS Gothic")

plt.ylabel("value")
plt.xlabel("Step")
# print parameta
Ymax=1.2; Ymin=-0.3
plt.ylim(Ymin,Ymax)
plt.xlim(0,knum)
xp=knum*1/5; yp=-Ymax*1/7  #plt.textの位置座標
#
plt.text(xp,yp, "g1={:.3g},g2={:.3g},g3={:.3g},g4={:.3g}".format(g[1],g[2],g[3],g[4]))
plt.text(xp,yp-Ymax/15, "g5={:.3g},g6={:.3g}".format(g[5],g[6]))


plt.legend(loc='center right') #labelの表示
plt.grid()
   # 表示
plt.show()  