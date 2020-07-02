def buildDocCollectionSimple(filename):
    lines = open(filename).read().splitlines()
    dic=dict()
    for i in range(len(lines)):
        words = lines[i].split(" ")
        if words[0]==".I":
            id=lines[i].replace(".I","").replace(" ","")
            dic[id]=""
        elif words[0]==".T":
            i+=1
            while (i<len(lines) and (".I" not in lines[i])):
                dic[id]+=lines[i]
                i+=1
            i-=1
    return dic
        
print(buildDocCollectionSimple("cacmShort-good.txt"))
