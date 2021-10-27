import os
import csv

f=open('../../../washington-square1.csv')
f1=open('train.txt','w')
fl = csv.reader(f)
fields = next(fl)
rows=[]
for row in fl:
    dic = {}
    for i in range(len(row)):
        dic[fields[i]] = row[i]
    rows.append(dic)
    
for r in rows:
    if (r['taken_on'].startswith('2016-07')):
        s=r['image'].split('/')
        f1.write("/data_1/washington-square/2/201607/"+s[len(s)-1]+'\n')
for r in rows:
    if (r['taken_on'].startswith('2016-05')):
        s=r['image'].split('/')
        f1.write("/data_1/washington-square/2/201605/"+s[len(s)-1]+'\n')
f.close()
f1.close()