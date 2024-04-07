#digital control 
#デジタル制御　高橋安人
#p134a 表7-1
#p70T4_g(z).pyを利用して、g(z)を求める。
#T=4の場合のPLOT
#L=3のため、T=4,L1=3、N=0とする。
#L=N*T+L1
#
#(4-11)G(z)に変換する。その時#(4-12)のパラメータ計算
#←これだと一寸違いが出る
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
np.set_printoptions(precision=5, suppress=True)

#(4-12),Fig 4-4から選んだパラメータ
T1=10;T2=6
T=4;L1=3;N=0 
L=N*T+L1

#case(a)
T1=10;T2=6;L1=3
#case(b)
#T1=11;T2=6.6;L1=3.3
#case(c)
#T1=9;T2=5.4;L1=2.7

#case(d) #shimojo 20%UP
#T1=12;T2=7.2;L1=3.6

#case(e) #shimojo　20%down
#T1=8;T2=4.8;L1=2.4

#case(f) #shimojo 30%UP
#T1=13;T2=7.8;L1=3.9

#case(g) #shimojo　30%down
#T1=7;T2=4.2;L1=2.1

print("\nN=",N,",  T1=",T1,",  T2=",T2,",  L=",L1)
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
#20240320 BGN
#a1=-1.241;a2=0.379
#b1=0.004;b2=0.108;b3=0.041
#20240320 END

print("\na1,a2 :",a1,a2) #;a_para=str(a1)+","+str(a2)
print("\nb1,b2,b3 :",b1,b2,b3) #;b_para=str(b1)+","+str(b2)+","+str(b3)
#20240320
#a1=-1.241;a2=0.379
#b1=0.004;b2=0.108;b3=0.041

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

#今回単位ステップの平衡値は”１”となる
# G(1)=g1+g2+....gn-1+gn/1-p
# p1=(G(1)-(g1+g2+....gn))/(G(1)-(g1+g2+....g(n-1))
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

#case(a)
#    i= 6 0.7078867001593836 0.7328960766436616 0.03532963181628793
#case(b)
#    i= 6 0.7368784811046724 0.7672343428446354 0.04119520724021669
#case(c)
#    i= 5 0.6886477822584309 0.7209511685634343 0.04690843002363843
    

##############################################################
#　                   グラフを描く(PLOT)                       #
###############################################################
t=np.arange(0,knum)
plt.plot(t,Y,'-or',label="y(k)")     #最終出力 
plt.plot(t,g*5.0,'-*b',label="g(k)x5")  #最終出力 ×5

strg1="T="+str(T)+",  T1="+str(T1)+",  T2="+str(T2)+",  L="+str(L1)
plt.title("図4-4 プロセス応答: "+strg1, fontname="MS Gothic")
plt.ylabel("Response y(k)")
plt.xlabel("Step")
# print parameta
Ymax=1.3; Ymin=-0.2
#plt.ylim(Ymin,Ymax)
plt.xlim(0,knum)

#
xp=knum*2/5; yp=Ymax*4/7  #plt.textの位置座標#
plt.text(xp,yp, "a1,a2={:.3g},{:.3g}".format(a1,a2) )
plt.text(xp,yp-Ymax/15, "b1,b2,b3={:.3g},{:.3g},{:.3g}".format(b1,b2,b3))

# print parameta
xp=knum*0.2/5; yp=Ymax*1/7  #plt.textの位置座標
plt.text(xp,yp, "g1={:.3g},g2={:.3g},g3={:.3g},g4={:.3g},g5={:.3g},g6={:.3g}".format(g[1],g[2],g[3],g[4],g[5],g[6]))
#plt.text(xp,yp-Ymax/15, "g5={:.3g},g6={:.3g}".format(g[5],g[6]))

plt.legend(loc='lower right') #labelの表示
plt.grid()
   # 表示
plt.show()  