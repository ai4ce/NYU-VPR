import math

ftest=open("image_name.txt","r")
f=open("match.txt","w")
ftrain=open("trainNorm.txt","r")
testn=[]
testx=[]
testy=[]
trainn=[]
trainx=[]
trainy=[]
for line in ftest:
    s=line.split()
    testn.append(s[0])
    testx.append(float(s[1]))
    testy.append(float(s[2]))
for line1 in ftrain:
    s=line1.split()
    trainn.append(s[0])
    trainx.append(float(s[1]))
    trainy.append(float(s[2]))
for i in range(len(testn)):
    minD=float("inf")
    trainname=""
    ind=0
    for j in range(len(trainn)):
        distance = math.sqrt(((float(trainx[j])-testx[i])**2)+((float(trainy[j])-testy[i])**2))
        if distance<minD:
            minD=distance
            ind=j
    f.write("/data_1/washington-square/2/train/"+testn[i]+" "+"/data_1/washington-square/2/train/"+trainn[ind]+"\n")
f.close()
ftest.close()
ftrain.close()