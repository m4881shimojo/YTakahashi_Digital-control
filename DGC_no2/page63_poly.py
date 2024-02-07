#digital control 
#デジタル制御　高橋安人
# 20240202 shimojo
#表4-3むだ時間を含む3連振動系のゼロ　
#ゼロ点をPLOTする
#zB1(z)+B2(z)の値は、page63v2.pyの出力結果-->Bza=np.array()
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
Bza=np.array([[0.,     -1.0915,  4.536,  -8.2534,  8.95,   -5.6703,  1.8504], #L1=0
              #
             [-0.2806,  0.3039,  1.4288, -4.1379,  5.5003, -3.9206,  1.4275], #L1=0.1
             [-0.5309,  1.5684, -1.4131, -0.3473,  2.3029, -2.2884,  1.0297], #L1=0.2
             [-0.7499,  2.6978, -3.9799,  3.1053, -0.6313, -0.7791,  0.6584], #L1=0.3
             #
             [-7.573,21.828,-22.159,5.463,5.982,-4.615,0.970], #L1=0.1書籍
             [-1.237,5.762,-10.002,7.470,-1.442,-.910,.359],   #L1=0.2書籍
             [-1.657,6.372,-9.509,6.662,-1.904,-0.048,0.087]]) #L1=0.3書籍
print("\npolinominal=zB1(z)+B2(z)\n",Bza)
##################################################
# Start Main routine
##################################################
#--------------------------------------------------
#--------------------------------------------------
#calculate ZERO  zB1(z)+B2(z)
#          G(z)=(zB1(z)+B2(z))/zA(z)
#--------------------------------------------------
#--------------------------------------------------
##################################################
#F(z)の極を求める
#

fig = plt.figure(figsize = (9,5),tight_layout=True) 
plt.suptitle("表4-3むだ時間を含む3連振動系のゼロ", fontname="MS Gothic")

plt.subplot(121,title="program") 
for k in range(0,4):   
    revD=Bza[k,:]   
    revD=revD[::-1] # 逆順表示
    pol=np.poly1d(revD,variable = "z") #n多項式関数の決定
    #pol.roots  #n多項式の根を求める関数 
    solAz=pol.r
    #print("\nk=",k,"\n",pol) #多項式の数式
    #print("\nzero =",solAz) #多項式の根　この場合はZERO
    
    if k==0:plot_color="kx"; msize=15;lab="L1=0"
    elif k==1:plot_color="c*"; msize=10;lab="L1=0.1"
    elif k==2:plot_color="r*"; msize=10;lab="L1=0.2"
    elif k==3:plot_color="b*"; msize=10;lab="L1=0.3"
    elif k==4:plot_color="c*"; msize=10;lab="L1=0.1"
    elif k==5:plot_color="r*"; msize=10;lab="L1=0.2"
    elif k==6:plot_color="b*"; msize=10;lab="L1=0.3"
            
    X=solAz.real;Y=solAz.imag
    #
    plt.plot(X, Y,  plot_color, markersize=msize,label=lab)
#
plt.ylabel("imag ")
plt.xlabel("real")
plt.legend(loc='upper left') #labelの表示
plt.grid()
plt.xlim(-7,3)
plt.ylim(-1,1)


plt.subplot(122,title="book") #　
for k in range(4,7):
    revD=Bza[k,:]  
    revD=revD[::-1] # 逆順表示
    pol=np.poly1d(revD,variable = "z") #n多項式関数の決定
    #pol.roots  #n多項式の根を求める関数 
    solAz=pol.r
    #print("\nk=",k,"\n",pol) #多項式の数式
    #print("\nzero =",solAz) #多項式の根　この場合はZERO
    
    if k==0:plot_color="yx"; msize=20;lab="L1=0"
    elif k==1:plot_color="c*"; msize=10;lab="L1=0.1"
    elif k==2:plot_color="r*"; msize=10;lab="L1=0.2"
    elif k==3:plot_color="b*"; msize=10;lab="L1=0.3"
    elif k==4:plot_color="c*"; msize=10;lab="L1=0.1"
    elif k==5:plot_color="r*"; msize=10;lab="L1=0.2"
    elif k==6:plot_color="b*"; msize=10;lab="L1=0.3"
            
    X=solAz.real;Y=solAz.imag
    #
    plt.plot(X, Y,  plot_color, markersize=msize,label=lab)
plt.ylabel("imag ")
plt.xlabel("real")
plt.legend(loc='upper left') #labelの表示#
plt.grid()
plt.xlim(-7,3)
plt.ylim(-1,1)

plt.show() 



