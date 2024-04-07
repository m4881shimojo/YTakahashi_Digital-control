#digital control 
#デジタル制御　高橋安人
#
#7.2節　LQ制御系の根軌跡
#p125 図7-1　#20231230
#ディジタルサーボ系の根軌跡を求める
#制御対象--->　p86　マスダンパ系(5-16)
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
# プラント
#G(z)=(az+b)/((z-1)(z-p))     Eq.7-8
#a=p+T-1, b=(1-p-pT)  Eq.7-8
#
#Fc(z)=z^4+a(1)z^3+a(2)z^2+a(3)z+a(4)   Eq.7-9
# a(1)=(ab-w(1+p)^2)/(wp)
# a(2)=(w(1+p*p+(1+p)^2) +a*a+b*b)/(wp)
# a(3)=(ab-w(1+p)^2)/(wp)
# a(4)=1

def root_log(wbeg,wend,dw,plot_color):
    n=7 #次数
    T=4 #sampling period
    T1=10;T2=6;r=T2/T1
    p1=np.exp(-T/T1);p2=np.exp(-T/T2)
    b1=1-(p1-r*p2)/(1-r)
    b2=p1*p2-(p2-r*p1)/(1-r)
    b3=p1*p2-(p2-r*p1)/(1-r)
    p3=(b1*p2+b2)/(1-p2)



#######

    #
    Az=np.zeros(n)
    tw=np.arange(wbeg,wend,dw)
    knum=len(tw)
    wt=np.zeros(knum)
    for i in range(knum):
        if tw[i]>0:wt[i]=10**tw[i]
        else: wt[i]=(1/10)**(-tw[i])
    #
    print(plot_color,"wbeg=",wbeg,wt[0])
    #
    for k in range(knum):
        w=wt[k]
        Az[0]=1
        Az[1]=-2-(p1+1/p1)-(p2+1/p2)
        Az[2]=3+2*(p1+1/p1)+2*(p2+1/p2)+(p1+1/p1)*(p2+1/p2)-b1*b2/(w*p1*p2)
        Az[3]=-4-2*(p1+1/p1)-2*(p2+1/p2)-2*(p1+1/p1)*(p2+1/p2)-(b1**2+b2**2)/(w*p1*p2)
        Az[4]=Az[2]
        Az[5]=Az[1]
        Az[6]=1
        pol=np.poly1d(Az) #n多項式関数の決定
        solAz=pol.r
        #AzC[k]=solAz
        X=solAz.real;Y=solAz.imag
        #plt.plot(X, Y, plot_color)
        ax1.plot(X, Y, plot_color)
        #end root_plot

##################################################
# Start Main routine
fig = plt.figure(figsize=(6, 6)) # Figureの初期化
#横ｘ縦（単位：inch）
ax1= plt.axes()

##################################################
        
wbeg=1;wend=5;dw=0.1;plot_color="c."       
root_log(wbeg,wend,dw,plot_color)

wbeg=0;wend=1;dw=0.1;plot_color="g."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-1;wend=0;dw=0.01;plot_color="r."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-2;wend=-1;dw=0.001;plot_color="b."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-3;wend=-2.;dw=0.001;plot_color="m."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-4;wend=-3;dw=0.001;plot_color="y."       
root_log(wbeg,wend,dw,plot_color)

#wbeg=-5;wend=-4;dw=0.01;plot_color="w."
wbeg=-7;wend=-4;dw=0.1;plot_color="r."        
root_log(wbeg,wend,dw,plot_color)

#wbeg=-8;wend=-5;dw=0.01;plot_color="k." 
wbeg=-8;wend=-7;dw=0.01;plot_color="k."       
root_log(wbeg,wend,dw,plot_color)


#############
# 2nd
#############


num=100
circle_x=np.zeros(num) 
circle_y=np.zeros(num) 

for k in range (0,num):
     rad=2*np.pi/num*k
     circle_x[k]=np.sin(rad);circle_y[k]=np.cos(rad)
     #plt.plot(np.sin(rad),np.cos(rad),'.k')
     
plt.plot(circle_x,circle_y,'--k')
#
strgp=""
Title_para=""
plt.title("図7-2 LQIプロセス制御系の根軌跡 :"+Title_para, fontname="MS Gothic")
#
plt.ylabel("imag ")
plt.xlabel("real")
#
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


#plt.xlim(-1.5,1.5)
#plt.ylim(-1.5,1.5)
limxy=1.5
plt.xlim(-limxy,limxy)
plt.ylim(-limxy,limxy)

plt.xlim(-6,2)
plt.ylim(-4,4)

plt.show() 


