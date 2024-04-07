#digital control 
#デジタル制御　高橋安人
#
#7.2節　LQ制御系の根軌跡
#p125 図7-1　#20231230
#20240312 見直し
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
##        
####################### Log plot #######################
## 
def root_log(wbeg,wend,dw,plot_color):
    n=5 #次数
    T=0.1;p=np.exp(-T)
    a=p+T-1; b=(1-p-p*T) 
    #
    Az=np.zeros(n)
    tw=np.arange(wbeg,wend,dw)
    knum=len(tw)
    wt=np.zeros(knum)
   
    for i in range(knum):
        wt[i]=10**tw[i]
    
    # wbeg<wend PLOTの始まりという場合、wを小さくして行く
    #イメージからはPLOTの始まりはwendとなる   
    print(plot_color,"wbeg=",wbeg,wt[0])
    ax1.plot(10, 10, plot_color,label="{:.3g}".format(10**wend))
    #10,10はplot範囲外の意味。100,100でもよい

    #
    for k in range(knum):
        w=wt[k]
        Az[0]=1
        Az[1]=(a*b-w*(1+p)**2)/(w*p)
        Az[2]=(w*(1+p*p+(1+p)**2) +a*a+b*b)/(w*p)
        Az[3]=(a*b-w*(1+p)**2)/(w*p)
        Az[4]=1
        pol=np.poly1d(Az) #n多項式関数の決定
        solAz=pol.r
        #AzC[k]=solAz
        X=solAz.real;Y=solAz.imag
        ax1.plot(X, Y, plot_color)
        #if knum==0: ax1.plot(X, Y, plot_color,marker='o',label="str(wt[0]")
        #else:ax1.plot(X, Y, plot_color)
        #end root_plot
  
##        
####################### Linear plot #######################
## 
def root_linear(wbeg,wend,dw,plot_color):
    n=5 #次数
    T=0.1;p=np.exp(-T)
    a=p+T-1; b=(1-p-p*T) 
    #
    Az=np.zeros(n)
    tw=np.arange(wbeg,wend,dw)
    #print(tw)
    knum=len(tw)
    #tw=np.zeros(knum)
    #    
    for k in range(knum):
        w=tw[k]
        Az[0]=1
        Az[1]=(a*b-w*(1+p)**2)/(w*p)
        Az[2]=(w*(1+p*p+(1+p)**2) +a*a+b*b)/(w*p)
        Az[3]=(a*b-w*(1+p)**2)/(w*p)
        Az[4]=1
        pol=np.poly1d(Az) #n多項式関数の決定
        solAz=pol.r
        #AzC[k]=solAz
        X=solAz.real;Y=solAz.imag
        #plt.plot(X, Y, plot_color)
        ax1.plot(X, Y, plot_color)
        #end root_plot

##################################################
# Start Main routine
fig = plt.figure(figsize=(8, 8)) # Figureの初期化
#横ｘ縦（単位：inch）
ax1= plt.axes()
##################################################
#seekp=-5.8059
##
# Log plot #
##
wbeg=2;wend=5;dw=0.1;plot_color="c."       
root_log(wbeg,wend,dw,plot_color)

wbeg=1.5;wend=2;dw=0.1;plot_color="g."       
root_log(wbeg,wend,dw,plot_color)

wbeg=1.0;wend=1.5;dw=0.02;plot_color="r."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-1;wend=1.0;dw=0.01;plot_color="b."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-3;wend=-1;dw=0.0002;plot_color="m."       
root_log(wbeg,wend,dw,plot_color)

wbeg=-5;wend=-3;dw=0.05;plot_color="y*"       
root_log(wbeg,wend,dw,plot_color)

wbeg=-10;wend=-5;dw=0.05;plot_color="k."       
root_log(wbeg,wend,dw,plot_color)

##
# Linear plot #
##
wbeg= 1.5e-06;wend= 1.6e-06;dw=1.e-09;plot_color="m."       
#root_linear(wbeg,wend,dw,plot_color)

#wbeg= 1.560982e-06;wend= 1.5609821e-06;dw=1.e-14;plot_color="k*"       
#root_linear(wbeg,wend,dw,plot_color)


#############
# Plot unit circle
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
plt.title("図7-1 ディジタルサーボ系の根軌跡 :"+Title_para, fontname="MS Gothic")
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


#狭い範囲
limxy=0.5
plt.xlim(-limxy,limxy)
plt.ylim(-limxy,limxy)
plt.xlim(0.5,1.2)
plt.ylim(-limxy,limxy)

#広い範囲
#plt.xlim(-7,2)
#plt.ylim(-4.5,4.5)

plt.legend(loc='lower right')
plt.show() 


