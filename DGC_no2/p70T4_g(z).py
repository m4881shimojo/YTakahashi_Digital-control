#digital control 
#デジタル制御　高橋安人
#p69図4-4をPLOTする　20231106 20240202
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
#
np.set_printoptions(precision=3, suppress=True)

#(4-12),Fig 4-4から選んだパラメータ
T1=10;T2=6
T=4;L1=3;N=0 #表4-5ではL=3-->N=0,L1=3
L=N*T+L1
#L=3
#K=1としてる
#G(z)=K(b1z^2+b2z+b3)/{z^(N+1)(z^2+s1z+a2)}
#N=0 --> z^(N+1) --> z^(1) 
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--


print("\nN,T,L,L1 :",N,T,L,L1)
#係数を(4-12)を用いて決める
#
p1=np.exp(-T/T1);p2=np.exp(-T/T2)
a1=-(p1+p2);a2=p1*p2
d1=np.exp(L1/T1);d2=np.exp(L1/T2);r=T2/T1#r NOT EQ "0"
b1=1-(p1*d1-r*p2*d2)/(1-r)
b2=(p1*d1*(1+p2)-r*p2*d2*(1+p1))/(1-r)-(p1+p2)
b3=p1*p2*(1-(d1-r*d2)/(1-r))
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
print("\na1,a2 :",a1,a2) #;a_para=str(a1)+","+str(a2)
print("\nb1,b2,b3 :",b1,b2,b3) #;b_para=str(b1)+","+str(b2)+","+str(b3)


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

###############################################################
#　                   グラフを描く(PLOT)                       #
###############################################################
t=np.arange(0,knum)
plt.plot(t,Y,'-or',label="y(k)")     #最終出力 
plt.plot(t,g*5.0,'-*b',label="g(k)x5")  #最終出力 ×5


plt.title("図4-4 プロセス応答 T="+str(T), fontname="MS Gothic")

plt.ylabel("Responce y(k)")
plt.xlabel("Step")
# print parameta
Ymax=1.2; Ymin=-0.3
plt.ylim(Ymin,Ymax)
plt.xlim(0,knum)
xp=knum*2/5; yp=-Ymax*1/7  #plt.textの位置座標
#
plt.text(xp,yp, "a1,a2={:.3g},{:.3g}".format(a1,a2) )
plt.text(xp,yp-Ymax/15, "b1,b2,b3={:.3g},{:.3g},{:.3g}".format(b1,b2,b3))
plt.legend() #labelの表示
plt.grid()
   # 表示
plt.show()  