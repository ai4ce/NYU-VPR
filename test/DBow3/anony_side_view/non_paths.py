import csv

def load_csv():
    rows=[]
    with open('/home/deansheng/washington-square1.csv') as f:
        fl=csv.reader(f)
        fields=next(fl)
        for row in fl:
            dic={}
            for i in range(len(row)):
                dic[fields[i]]=row[i]
            rows.append(dic)
    return rows

def find_path(image, rows):
    for r in rows:
        if image.split('/')[-1] in r['image']:
            path="/data_1/washington-square/" + r['image']
            break
    return path

f=open('train_image_paths.txt')
f1=open('train_origin_image_paths.txt','w')
rows=load_csv()
for line in f:
    f1.write(find_path(line[:-1],rows)+'\n')
