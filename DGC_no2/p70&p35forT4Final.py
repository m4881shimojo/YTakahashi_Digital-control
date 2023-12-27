#digital control 
#デジタル制御　高橋安人
#p69図4-4をPLOTする　20231106
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


#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常
#z(^-1)が掛かっていると考えると
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-2]+b2*U[k-3]+b3*U[k-4]) 
Y[0]=0.0
Y[1]=-a1*Y[0]#+(b1*U[-1]+b2*U[-2]+b3[-3])  
Y[2]=-(a1*Y[1]+a2*Y[0])+(b1*U[0])#+b2*U[-1]+b3[-2])  
Y[3]=-(a1*Y[2]+a2*Y[1])+(b1*U[1]+b2*U[0])#+b3[-1])  
#Y[4]=-(a1*Y[3]+a2*Y[2])+(b1*U[2]+b2*U[1]+b3*U[0]) 


for k in range(4,knum):   
    #Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]) 
    #Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-3]+b2*U[k-4]+b3*U[k-5])
    Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-2]+b2*U[k-3]+b3*U[k-4])  
    #print(k,Y[k])
#
g[0]=Y[0]
for j in range(1,knum):
    g[j]=(Y[j]-Y[j-1])
print("\ng=",g)

#　グラフを描く
t=np.arange(0,knum)
plt.plot(t,Y,'-or')     #最終出力 
plt.plot(t,g*5.0,'-*b')  #最終出力 ×5


plt.title("図4-4 プロセス応答 T="+str(T), fontname="MS Gothic")
Ymax=1.2; Ymin=-0.3
plt.ylim(Ymin,Ymax)
plt.xlim(0,knum)
plt.ylabel("Responce y(k)")
plt.xlabel("Step")
xp=knum*2/5; yp=-Ymax*1/7  #plt.textの位置座標
#plt.text(xp,yp, "z1={:.3g}".format(z1) )
plt.text(xp,yp, "a1,a2={:.3g},{:.3g}".format(a1,a2) )
plt.text(xp,yp-Ymax/15, "b1,b2,b3={:.3g},{:.3g},{:.3g}".format(b1,b2,b3))

   # 表示
plt.show()  