#digital control 
#デジタル制御　高橋安人
#fig1.3 shimojo 20231025
#
import numpy as np
import matplotlib.pyplot as plt
#
#print("start")
G=np.zeros(11); U=np.zeros(11); Y=np.zeros(11); #0,1,....8,9,10　の11個
UG=np.zeros([11,11]); YY=np.zeros([11,11]);P_para=np.zeros([3])
Num=11
# input parameter
P=-0.5; R=1.0; G[1]=1.
P_para=-0.5,0.7,1.07,-1.07
#print(P_para)

for m in range(4):
    P=P_para[m]

#単位パルス入力に対する応答波形ｇ
    for i in range(2,Num):
        G[i]=P*G[i-1]
        Y[i]=Y[i-1]
    #print(G)
    t=np.arange(0,Num)
    #G,U,Y
    Kn=1
    for l in range(0,Kn):
        #U=np.zeros(11);Y=np.zeros(11)
        U[l]=1.0
        #while 
        for k in range(0,Num):
            #Y[k]=y0
            y0=0.0
            
            for j in range(0,k):
                y0=y0+U[j]*G[k-j]
            UG[l,k]=y0 
            #print(k,UG[l,k]) 
            #
        if m==0 :plt.plot(t,UG[l,],'--*k')
        if m==1 :plt.plot(t,UG[l,],'--*r')
        if m==2 :plt.plot(t,UG[l,],'--*b')
        if m==3 :plt.plot(t,UG[l,],'--*c')
        plt.plot(t,U, drawstyle='steps-post',color='g', linestyle='solid', marker='o')

#
#plt.plot(t,Y,'--*r')

plt.ylim(-2.,2.)
plt.xlim(0,11)
plt.title("g(k), P="+str(P_para))
plt.ylabel("Responce g(k)")
plt.xlabel("step k")
   # 表示
plt.show() 





#%matplotlib inline

   # 表示
plt.show()    
