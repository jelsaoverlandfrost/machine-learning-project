
def protoemission(listy, listx):
    print(listy)
    print(listx)
    dictionary = {}
    ydictionary = {}
    for i in range(len(listy)):
        if listx[i] not in dictionary:
            newdic = {listy[i]: 1.0}
            dictionary[listx[i]] = newdic
        if listy[i] not in dictionary[listx[i]]:
            dictionary[listx[i]][listy[i]] = 1.0
        else:
            dictionary[listx[i]][listy[i]] = dictionary[listx[i]][listy[i]] + 1
        if listy[i] not in ydictionary:
            ydictionary[listy[i]] = 1.0
        else:
            ydictionary[listy[i]] = ydictionary[listy[i]] + 1.0

    return dictionary, ydictionary


def finalemission(dic, ydic, x, y, k):
    if x not in dic:
        return k / (ydic[y] + k)
    else:
        if y not in dic[x]:
            return 0
        else:
            return dic[x][y] / (ydic[y] + k)


def testemission(listy, listx, y, x, k):
    counter = 0.0
    ycounter = 0.0
    for i in range(len(listy)):
        if (listx[i] == x) & (listy[i] == y):
            counter = counter + 1
        if (i == y):
            ycounter = ycounter + 1
    #    print(y,counter,intercount(listy,y)+k)
    if counter != 0:
        return float(counter) / (ycounter + k)
    else:
        return k / (ycounter + k)


def argmaxx(listy, listx, x, k, rangey):
    #    rangey=["O","B-positive","B-neutral","B-negative","I-positive","I-neutral","I-negative"]
    maxarg = 0
    tag = ""
    for i in rangey:
        #        print(i,testemission(listy,listx,i,x,k))
        if testemission(listy, listx, i, x, k) > maxarg:
            maxarg = testemission(listy, listx, i, x, k)
            tag = i
    return tag, maxarg


def settle(filename):
    print("Start with training")
    file = open(filename, "r", encoding="UTF-8")
    listy = []
    listy = []
    listx = []
    for line in file:
        listline = line.split()
        if len(listline) != 0:
            listx.append(line[:(len(line) - len(listline[-1])) - 2])
            #            listx.append(listline[0].lower())
            listy.append(listline[-1])
    print("Done with training")
    #    print(listx)
    return listx, listy


def testproto(testname, trainname, result, k):
    try:
        testfile = open(testname, "r", encoding="UTF-8")
        resultfile = open(result, "w+", encoding="UTF-8")
        listx, listy = settle(trainname)
        predict = {}
        number = 0
        for word in testfile:
            listword = word.split()
            if (len(listword) == 1):
                #                newword=listword[0].lower()
                #                if (newword not in predict):
                if (listword[0] not in predict):
                    (key, value) = argmaxx(listy, listx, listword[0], k)
                    predict[listword[0]] = key
                resultfile.write(listword[0] + " " + predict[listword[0]] + "\n")
            #                    (key,value)=argmaxx(listy,listx,newword,k)
            #                    predict[newword]=key
            #                resultfile.write(listword[0]+" "+predict[newword]+"\n")
            else:
                resultfile.write("\n")
            number = number + 1
            print(number)
    #        print(predict)
    finally:
        resultfile.close()


def testword(word, trainname, k):
    listx, listy = settle(trainname)
    #    print (intercount(listy,"O"))
    #    print(word.lower())
    #    print(argmaxx(listy,listx,word.lower(),k))
    return 0


def unseennumber(testname, train):
    test = open(testname, "r", encoding="UTF-8")
    listx, listy = settle(train)
    counter = 0
    for word in test:
        listword = word.split()
        if (len(listword) == 1):
            if (listword[0].lower() not in listx):
                counter = counter + 1
    return counter


def transition(listy, first, second):
    check = False
    numerator = 0.0
    denominator = 0.0
    if (first == ""):
        check = True
        denominator = 1
    for line in listy:
        if ((line == second) & check):
            numerator = numerator + 1
        #            print(check,numerator)
        check = False

        if (line == first):
            check = True
            denominator = denominator + 1
    #            print(check,denominator)

    if (second == ""):
        numerator = numerator + 1

    if (denominator != 0):
        return numerator / denominator
    else:
        return 0


def transitionlist(listy, rangexy):
    dictionary = {}
    rangey = rangexy.copy()
    rangey.append("")
    for i in range(len(rangey)):
        anotherdictionary = {}
        for j in range(len(rangey)):
            if ((i + j != 2 * len(rangey))):
                anotherdictionary[rangey[j]] = transition(listy, rangey[i], rangey[j])
        dictionary[rangey[i]] = anotherdictionary
    return dictionary


def handle(filename):
    print("Start with y")
    file = open(filename, "r", encoding="UTF-8")
    listy = []
    for line in file:
        listline = line.split();
        if (len(listline) != 0):
            listy.append(listline[-1])
        else:
            listy.append("")
    print("Done with y")
    return listy


# def argmaxy(listy,y):
#    rangey = ["O","B-positive","B-neutral","B-negative","I-positive","I-neutral","I-negative",""]
#    maxarg = 0
#    tag = ""
#    for word in rangey:
#        if (transition(listy,y,word)>maxarg):
#            maxarg = transition(listy,y,word)
#            tag = word
#    return (tag,maxarg)


# listy=handle(train)
# print(transition(listy,"B-negative","O"))
# print(argmaxy(listy,"B-positive"))


def extractx(filename):
    print("Start with x")
    file = open(filename, "r", encoding="UTF-8")
    listx = []
    for line in file:
        listline = line.split();
        if (len(listline) != 0):
            listx.append(line[:-1])
        #            listx.append(listline[0].lower())
        else:
            listx.append("")

    print("Done with x")
    return listx


class node:
    def __init__(self, prob, parent):
        self.prob = prob
        self.parent = parent


# class prob:
#    def __init__(self,number,magnitude):
#        self.number=number
#        self.mag=magnitude
#    def multiply(self,other.)
def ytag(trainname):
    file = open(trainname, "r", encoding="UTF-8")
    listy = []
    for line in file:
        listline = line.split();
        if (len(listline) == 2):
            if (listline[1] not in listy):
                listy.append(listline[1])
    return listy


def viterbi(testname, trainname, resultname, k):
    #    Initialization

    try:
        resultfile = open(resultname, "w+", encoding="UTF-8")
        current = []
        prob = []
        emissionx, emissiony = settle(trainname)
        edic, ydic = protoemission(emissiony, emissionx)
        x = extractx(testname)
        rangey = ytag(trainname)
        ydictionary = transitionlist(handle(trainname), rangey)
        where = 0;
        for wordie in x:
            # Initialise newlist to store prob
            where = where + 1
            if (where % 100 == 0):
                print(str(where))
            #            print(current)
            newlist = []
            argmax = 0
            tag = 0
            # if start of sentence
            if (len(prob) == 0):
                for y in rangey:
                    probability = ydictionary[""][y]
                    word = node(probability, 7)
                    newlist.append(word)
                prob.append(newlist)

            # if not start of sentence or end of sentence
            elif (wordie != ""):
                latestprob = prob[len(prob) - 1]
                for i in range(len(rangey)):
                    argmax = 0
                    for j in range(len(rangey)):
                        initialprob = latestprob[j].prob
                        transitionprob = ydictionary[rangey[j]][rangey[i]]
                        emissionprob = finalemission(edic, ydic, wordie, rangey[i], k)
                        #                        emissionprob = finalemission(edic,ydic,wordie.lower(),rangey[i],k)
                        #                        print(initialprob,transitionprob,emissionprob,argmax)
                        a = initialprob * transitionprob * emissionprob
                        if (a > argmax):
                            argmax = a
                            tag = j
                    word = node(argmax, tag)
                    #                    print(argmax,tag)
                    newlist.append(word)
                prob.append(newlist)
                # If end of sentence
            else:
                for j in range(len(rangey)):
                    initialprob = prob[len(prob) - 1][j].prob
                    transitionprob = ydictionary[rangey[j]][""]
                    a = initialprob * transitionprob
                    if (a > argmax):
                        argmax = a
                        tag = j
                currentpointer = tag
                position = -1
                order = []
                #                print(position,currentpointer)
                #                print(len(prob))
                #
                #                for i in prob:
                #                    print("!")
                #                    for j in i:
                #                        print(j.prob)

                #                print(current)
                while (position != ((-1 * len(prob)) - 1)):
                    order.append(currentpointer)
                    currentpointer = prob[position][currentpointer].parent
                    position = position - 1
                #                print(len(current),len(order))
                #                print(order)
                for i in range(len(order)):
                    resultfile.write(str(current[i]) + " " + rangey[order[(-1 * i) - 1]])
                    resultfile.write("\n")
                resultfile.write("\n")
            if (wordie != ""):
                current.append(wordie)
            else:
                newlist = []
                current = []
                prob = []
        # Out of loop
        #        for j in range(7):
        #            initialprob = prob[len(prob)-1][j].prob
        #            transitionprob = ydictionary[rangey[j]][""]
        #            if (initialprob*transitionprob>argmax):
        #                argmax=initialprob*transitionprob
        #                tag=j
        #        currentpointer=tag
        #        position=-1
        #        order=[]
        #                print(position,currentpointer)
        #        if (len(prob)!=0):
        #            while (position != ((-1*len(prob))-1)):
        #                order.append(currentpointer)
        #                currentpointer=prob[position][currentpointer].parent
        #                position=position-1
        ##                    print(len(current),len(order))
        ##                    print(order)
        #            for i in range(len(order)):
        #                resultfile.write(str(current[i])+" "+rangey[order[(-1*i)-1]])
        #                resultfile.write("\n")
        resultfile.write("\n")
    finally:
        resultfile.close()

viterbi('data/SG/dev.in', 'data/SG/train', 'data/SG/dev.p3.out', 1)