import os
import csv

f=open('../../washington-square1.csv')
f1=open('train.txt','w')
f2=open('train_image_paths.txt')
fl = csv.reader(f)
fields = next(fl)
rows=[]
for row in fl:
    dic = {}
    for i in range(len(row)):
        dic[fields[i]] = row[i]
    rows.append(dic)
    
for line in f2:
    s=line.split('/')
    imagen=s[len(s)-1][:-1]
    for r in rows:
        s1=r['image'].split('/')
        if (s1[len(s)-1]==imagen):
            f1.write(s1[len(s)-1]+' '+r['snapped_lat']+' '+r['snapped_lon']+'\n')
            break
f.close()
f1.close()
f2.close()