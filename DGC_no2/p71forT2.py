#digital control 
#デジタル制御　高橋安人
#p71の結果を検証する
#p69図4-4をPLOTする　20231106
#(4-10)G(s)の応答を算出する
#L=3のため、T=2,L1=1とする。
#L=3のため、T=4には不可
#その場合、N=0,L1=3とする
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
T=2;L1=1;K=1;N=1 #表4-5ではL=3-->N=1,L1=1
L=N*T+L1
#L=3
#G(z)=K(b1z^2+b2z+b3)/{z^(N+1)(z^2+s1z+a2)}
#N=1 --> z^(N+1) --> z^(2) 
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常
#分子に、z(^-2)が掛かっていると考えると
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-3]+b2*U[k-4]+b3*U[k-5]) 

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


#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) <--これが通常
#z(^-2)が掛かっていると考えると
#Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-3]+b2*U[k-4]+b3*U[k-5]) 

Y[0]=0.0
Y[1]=-a1*Y[0]#+(b1*U[-2]+b2*U[-3]+b3[-4])  
Y[2]=-(a1*Y[1]+a2*Y[0])#+(b1*U[-1]+b2*U[-2]+b3[-3])
Y[3]=-(a1*Y[2]+a2*Y[1])+(b1*U[0])#+b2*U[-1]+b3[-2]) 
Y[4]=-(a1*Y[3]+a2*Y[2])+(b1*U[1]+b2*U[0])#+b3[-1]) 
#Y[5]=-(a1*Y[4]+a2*Y[3])+(b1*U[2]+b2*U[1]+b3*U[0])  

for k in range(5,knum):   
    #Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2]+b3*U[k-3]) 
    Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-3]+b2*U[k-4]+b3*U[k-5]) 
    #print(k,Y[k])
#
g[0]=Y[0]
g[1]=Y[1]
for j in range(2,knum):
    g[j]=(Y[j]-Y[j-1])
    #print("\ng=",j,g)

############################################
#　　　　　4.4 時系列からの伝達関数　　　　   #
#             page 68                      #
############################################
# G(1)=g1+g2+....gn-1+gn/1-p
# p=(gn+1)/gn
# p2=gn+1/gn  e=|p2-p1|/p1
# e<0.05に達すればOK　その時のp1(or p2) をpとする

print("\ng=",g)
print("\n")
#
gsum=0.0
G1=0.0
p1_hat_old=g[5+1]/g[5]
#
#
for pnum in range(5,knum-1):
    gsum=0.0
    for j in range(1,pnum):
         gsum=gsum+g[j]

    G1=(gsum-g[pnum])+g[pnum]/(1-g[pnum]/g[pnum-1]) #p=g[pnum]/g[pnum-1]
    #print("G1",G1)
    #G1=1 #理論値
    #
    p2=g[pnum+1]/g[pnum]
    p1_hat=(G1-gsum)/(G1-(gsum-g[pnum]))
    ####
    #G1=(gsum-g[pnum-1])+g[pnum]/(1-p1_hat)
    #p1_hat=(G1-gsum)/(G1-(gsum-g[pnum]))
    ####
    #    
    e_1=abs(p1_hat-p2)/p1_hat
    e_2=p1_hat_old/p1_hat
    p1_hat_old=p1_hat
    G1=0.0
    print("pnum={:.3g}, e_1={:.5g}, e_2={:.5g}, p={:.5g}".format(pnum,e_1,e_2,p1_hat))    
    
# 単位ステップ応答の実測値からのG(z)推定
p1_hat=0.860 #書籍と同じとした
Dnum=8 #書籍と同じとした
bj=np.zeros(knum)
n_begin=2
bj[0]=g[n_begin]

for j in range(1,Dnum):
    bj[j]=g[n_begin+j]-p1_hat*g[n_begin+j-1]
print(bj)    

#　グラフを描く
t=np.arange(0,knum)
#plt.plot(t,Y,'-or')     #最終出力 
plt.plot(t,g,'-*r')     #最終出力 
plt.plot(t,bj,'-*b') 

#plt.title("図4-4 プロセス応答 T="+str(T), fontname="MS Gothic")
#Ymax=1.2; Ymin=-0.3
#plt.ylim(Ymin,Ymax)
#plt.xlim(0,knum)
#plt.ylabel("Responce y(k)")
#plt.xlabel("Step")
#xp=knum*2/5; yp=-Ymax*1/7  #plt.textの位置座標

#plt.text(xp,yp, "a1,a2={:.3g},{:.3g}".format(a1,a2) )
#plt.text(xp,yp-Ymax/15, "b1,b2,b3={:.3g},{:.3g},{:.3g}".format(b1,b2,b3))

   # 表示
plt.show()  