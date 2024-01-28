# digital control
# デジタル制御　高橋安人
# 図3-2ステップ応答に及ぼすゼロの影響　page.35
# shimojo 20231015
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#from numpy.linalg import inv
#
knum = 11
for l in range(4):
    Y = np.zeros(knum)
    U = np.ones(knum)

    a1 = -1.2
    a2 = 0.32
    
    if l==0:  z1=0.836; b1 = (1.+a1+a2)/(1-z1);b2 = -z1*b1
    if l==1:  z1=1.6; b1 = (1.+a1+a2)/(1-z1);b2 = -z1*b1
    if l==2:  z1=0.6; b1 = (1.+a1+a2)/(1-z1);b2 = -z1*b1
    if l==3:  b1 =0;b2=1.+a1+a2

 
    # z1 parameter z1=-b2/b1
    #z1 = 0.836  # (d)
    # z1=1.6 #(c)
    # z1=0.6 #(a)
    #b1 = (1.+a1+a2)/(1-z1)  # (3-13)より
    #b2 = -z1*b1
    # z1が存在しない場合？
    # b1=0;b2=1.+a1+a2;z1=999999 #(b)


    Y[1] = -a1*Y[0]+(b1*U[0])
    Y[2] = -(a1*Y[1]+a2*Y[0])+(b1*U[1]+b2*U[0])

    for k in range(2, knum):
        Y[k] = -(a1*Y[k-1]+a2*Y[k-2])+(b1*U[k-1]+b2*U[k-2])
        # print(k,Y[k])
    #
    #　グラフを描く
    t = np.arange(0, knum)
    #plt.plot(t, Y, '--or')  # 最終出力
    if l==0: plt.plot(t, Y, '--ok') 
    if l==1: plt.plot(t, Y, '--or') 
    if l==2: plt.plot(t, Y, '--og') 
    if l==3: plt.plot(t, Y, '--ob') 

plt.title("図3-2 ステップ応答に及ぼすゼロの影響", fontname="MS Gothic")
Ymax = 1.2
Ymin = -0.3
plt.ylim(Ymin, Ymax)
plt.xlim(0, knum)
plt.ylabel("Responce y(k)")
plt.xlabel("Step")
xp = knum*3/8
yp = Ymax*0/8  # plt.textの位置座標
#plt.text(xp, yp, "z1={:.3g}".format(z1))
plt.text(xp, yp, "z1=0.6(緑),1.6(赤),0.836(黒),存在しない(青)", fontname="MS Gothic")
# 表示
plt.show()
