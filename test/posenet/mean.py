import math
import utm

f=open("train.txt","r")
f1=open("trainNorm.txt","w")
count=0
sumx=0.0
sumy=0.0
count=0
ur=[]
for line in f:
    s=line.split()
    if (len(s)<=1):
        continue
    lat=float(s[1])
    lon=float(s[2])
    ures=utm.from_latlon(lat,lon)
    ur.append([ures[0],ures[1],s[0]])
    sumx+=ures[0]
    sumy+=ures[1]
    count+=1

mean = (sumx/count, sumy/count)
print(mean)


dx=[]
dy=[]
ex=0
ey=0
for ures in ur:
    dx.append(ures[0]-mean[0])
    dy.append(ures[1]-mean[1])
    normx=(ures[0]-mean[0])/559
    normy=(ures[1]-mean[1])/485
    f1.write(ures[2]+' '+str(round(normx,6))+' '+str(round(normy,6))+'\n')
    if normx>1 or normx <-1:
        ex+=1
    if normy>1 or normy <-1:
        ey+=1
    
f.close()
f1.close()