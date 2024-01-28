# matrix enshuu 20231023 shimojo
#Digital Control P47 
#p44に基づき、(a)(b)(c)の各ケースの伝達関数を求め応答を計算する
#Y　過去4回分の出力、YY：時系列データ、Y0：working

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA # Linear algebra
#

n=4 #4次系
knum=120 #サンプル数
#A,Bは伝達関数の係数
A=np.zeros(n);B=np.zeros(n) #a1,a2,a3,a4,b1,b2,b3,b4
U=np.zeros((n,1));Y=np.zeros((n,1))
YY=np.zeros(knum)# plotのデータアレイ

#Y0=np.zeros(4) #Working memory
r1=0.99; r2=1.0
mdeg1=5.0;mdeg2=45.0 #degree
a1=np.cos(2*np.pi/360*mdeg1)*r1
b1=np.sin(2*np.pi/360*mdeg1)*r1
a2=np.cos(2*np.pi/360*mdeg2)*r2
b2=np.sin(2*np.pi/360*mdeg2)*r2
#
#memo: インデントの追加は「Ctrl + ]」

for l in range(3):
    U=np.zeros((4,1));Y=np.zeros((4,1)) #reset
    
# ケース分けのパラメータ設定
    if l==0: q1=0.;q2=1.;c1=1.;c2=0. #Case (a)
    if l==1: q1=1.;q2=0.;c1=1.;c2=0. #Case (b)
    if l==2: q1=0.;q2=1.;c1=0.;c2=1. #Case (c)
    if l==3: q1=1.;q2=1.;c1=1.;c2=0. #Case (d)
#Case (d)は書籍に無い。for l in range(4):　とすれば描画

    #伝達関数を A,B　より作る
    bcq1=b1*c1*q1; bcq2=b2*c2*q2
    R1=a1*a1+b1*b1; R2=a2*a2+b2*b2 #今回は、R1=r1,R2=r2となりますが、
    A=np.array([-2*(a1+a2), (R1+R2+4*a1*a2),-2*(a1*R2+a2*R1),R1*R2])
    B=np.array([0,(bcq1+bcq2),-2*(a2*bcq1+a1*bcq2),(R2*bcq1+R1*bcq2+b1*b2*c1*q2)])

    #応答の算定　（伝達関数）
    y0=0.
    for k in range(0,knum):
        for j in range(n-1,0,-1):
            U[j]=U[j-1]
            Y[j]=Y[j-1]
        U[0]=1. # impulse input
        Y[0]=y0 #y0 <---y(k-1)
        y0=B.dot(U)-A.dot(Y) # (3-11)
        YY[k]=y0 #YY　はplotのため用意した
        #print(k,y0)
    #enk K loop
        
    #
    t=np.arange(0,knum)
    if l==0: plt.plot(t, YY, '--*b',label="(a)") 
    if l==1: plt.plot(t, YY, '--*r',label="(b)") 
    if l==2: plt.plot(t, YY, '--*g',label="(c)") 
    if l==3: plt.plot(t, YY, '--*k') 
    # End for loop (for l in range(3):)

# 以下はグラフを描くコマンド
plt.title("図3-4 2重信号系のステップ応答", fontname="MS Gothic")
Ymax = 25
Ymin = 0
plt.ylim(Ymin, Ymax)
plt.xlim(0, knum)
#xp = knum*4/8
#yp = Ymax*7/8  # plt.textの位置座標
#plt.text(xp, yp, "case(a)黒,case(b)赤,case(c)緑", fontname="MS Gothic") 
plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")
plt.grid()
plt.legend() #labelの表示
    # 表示
plt.show() 
        
