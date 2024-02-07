#digital control 
#デジタル制御　高橋安人
#p72
#g[i]と状態空間　(4-20)
#g[i]計算のため、 p70T4new.pyを利用
#L=3のため、N=0,T=4,L1=3とする。
#
#(4-11)G(z)に変換する。その時#(4-12)のパラメータ計算
#
#応答計算はp72
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
# 以上g[i]を求めるための計算
# g[i]が分かっていれば不用です

############################################
#　　　　　4.4 時系列からの状態空間形　　　  #
#             page 72                      #
############################################
p1_hat=0.7078867 #書籍とほぼ同じ
Dnum=6 #書籍と同じ
 
P=np.array([[0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,0.,0.,0.,p1_hat]])
q=np.array([[g[1]], #p72に説明
            [g[2]],
            [g[3]],
            [g[4]],
            [g[5]],
            [g[6]]])
c=np.array([1.,0.,0.,0.,0.,0.]) #


n=6
u=np.ones(knum) #step input
Xk1=np.zeros((n,1))
Xk=np.zeros((n,1))
Y=np.zeros(knum)# plotのデータアレイ
  #応答の算定　（X(k+1)=PX(k)+qu(K)）
y0=0.
for k in range(0,knum):
    Xk1=np.dot(P,Xk)+u[k]*q   
    Y[k]=np.dot(c,Xk1)
    Xk=np.copy(Xk1)    
#enk K loop


###############################################################
#　                   グラフを描く(PLOT)                       #
###############################################################
t=np.arange(0,knum)
plt.plot(t,Y,'-*r',label="Y(k)")     #最終出力 
plt.plot(t,g*5,'-*b',label="g(j)x5") 

#plt.plot(t,Y,'-or')     #最終出力 
#plt.plot(t,g,'-*r')     #最終出力 
#plt.plot(t,bj,'-*b') 
plt.title("図4-4 時系列からの状態空間形を用いた応答, T="+str(T), fontname="MS Gothic")

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