#digital control 
#デジタル制御　高橋安人
#list 2.3 page 24. shimojo 20231015
#調和振動　（共役複素極 p1,p2= α±jβ ）
#2次系の応答（リカーシブアルゴリズム）
#
import numpy as np
import matplotlib.pyplot as plt

#調和振動の例を示す。パラメータは(2-35)式を参考に適当に定めた
#Q1;Q2;C1;C2によって応答の振幅と位相が決まる

X0=0.;X1=0.;X2=0.
Q1=1.;Q2=1.
C1=1.;C2=1.;D=0.
D=0.

#(2-33)を基に以下のパラメータは適当に定めた
R=1.0 #(2-33) 例えばR=0.99では減衰振動となる
wT=2*np.pi*1/12 #radian
#
#set parameter （共役複素極 p1,p2= α±jβ ）
P1=R*np.cos(wT);P2=R*np.sin(wT);P3=-P2;P4=P1 # #(2-33)(2-35)

# impulse応答を計算
# input parameter
N=36
#N=120
U=np.zeros(N); #U=0.0
#U=np.ones(N); #U=0.0
U[0]=1. # impulse応答

#list 2-3 (page24)
for k in range(0,N):
     Y=C1*X1+C2*X2+D*U[k]     
     X0=P1*X1+P2*X2+Q1*U[k]
     X2=P3*X1+P4*X2+Q2*U[k]
     X1=X0
     plt.plot(k,Y,"ro")

# plt.show()で画面に表示（Jupyter Notebookの場合は不要）


Ymin=-3;Ymax=3
plt.ylim(Ymin,Ymax)
plt.xlim(0,N)
plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")

#print parameters

strg1="P1={:.3g},P2={:.3g},P3={:.3g},P4={:.3g},R={:.3g}".format(P1,P2,P3,P4,R)
#xp=N/5; yp=Ymax*3/5  #plt.textの位置座標
xp=N/5; yp=Ymax*3/4  #plt.textの位置座標
plt.text(xp,yp, "C1={},C2={},Q1={},Q2={},D={}".format(C1,C2,Q1,Q2,D) )
plt.title("List 2-3 2次系の応答:"+strg1, fontname="MS Gothic")

plt.show()
