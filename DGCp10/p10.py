#digital control 
#デジタル制御　高橋安人
#list 1.1 shimojo 20160728
#
import numpy as np
import matplotlib.pyplot as plt
#Dead beat response of robo arm
print("start")

G=np.zeros(11); U=np.zeros(11); Y=np.zeros(11); #0,1,....8,9,10　の11個
UG=np.zeros([11,11]); YY=np.zeros([11,11]);

# input parameter
P=-0.5; R=1.0; G[1]=1.
#G,U,Y
#単位パルス入力に対する応答波形ｇ
for i in range(2,11):
    G[i]=P*G[i-1]
 
#G,Y   
#応答出力がR=1となるように、入力Uを定める   
for k in range(1,11):
    U[k-1]=(R-Y[k])/G[1]
    for j in range(k,11,1):  
        Y[j]+=U[k-1]*G[j-k+1]
        UG[k-1,j]=U[k-1]*G[j-k+1] #add shimojo
        #YY[k,j]=Y[j]
    #print(k,U[k-1],Y[k]) #これは書籍Original

    #次からはプロットするためshimojoが追加
    #YY[1]=UG[0]
    #YY[2]=UG[0]+UG[1]
        ##YY[2]=YY[1]+UG[1]
    #YY[3]=UG[0]+UG[1]+UG[2]
        ##YY[3]=YY[2]+UG[2]
    #YY[4]=UG[0]+UG[1]+UG[2]+UG[3]
        ##YY[4]=YY[3]+UG[3]
#応答波形は単位応答波形のsum
for i in range(1,10):
    YY[i]=YY[i-1]+UG[i-1]

#%matplotlib inline
t=np.arange(0,11)
plt.ylim(-1.,2.)
plt.xlim(0,11)
plt.title("dead-beat respose")
plt.plot(t,YY[1],'--k')
plt.plot(t,YY[2],'--r')
plt.plot(t,YY[3],'--g')
plt.plot(t,YY[4],'--b')
plt.plot(t,YY[5],'--k')
plt.plot(t,YY[6],'--r')
plt.plot(t,YY[7],'--g')
plt.plot(t,YY[8],'--b')
plt.plot(t,YY[9],'--k')
plt.plot(t,YY[10],'--r')
plt.plot(t,Y,'-ok',label="Y")     #最終出力 label??

plt.ylabel("Responce")
plt.xlabel("sample time")
   # 表示
plt.show()    
