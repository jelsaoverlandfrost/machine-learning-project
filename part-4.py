import argparse
from collections import defaultdict

def protoemission(listy, listx):
    dictionary = {}
    ydictionary = {}
    for i in range(len(listy)):
        if listx[i] not in dictionary:
            #            dictionary[listx[i]][listy[i]]=1.0
            newdic = {}
            newdic[listy[i]] = 1.0
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
    if counter != 0:
        return float(counter) / (ycounter + k)
    else:
        return k / (ycounter + k)


def argmaxx(listy, listx, x, k, rangey):
    #    rangey=["O","B-positive","B-neutral","B-negative","I-positive","I-neutral","I-negative"]
    maxarg = 0
    tag = ""
    for i in rangey:
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
    finally:
        resultfile.close()


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

def secondtransition(listy, first, second, third):
    numerator = 0.0
    denominator = 0.0
    for line in range(len(listy)):
        if (line == 0):
            if ((first == "") & (second == "")):
                denominator = denominator + 1
                if (listy[line] == third):
                    numerator = numerator + 1
        #                    print(line)
        elif (line == 1):
            if ((first == "") & (listy[line - 1] == second)):
                denominator = denominator + 1
                if (listy[line] == third):
                    numerator = numerator + 1
        #                    print(line)
        elif (listy[line - 1] == ""):
            if ((first == "") & (listy[line - 1] == second)):
                denominator = denominator + 1
                if (listy[line] == third):
                    numerator = numerator + 1
        #                    print(line)
        else:
            if ((listy[line - 2] == first) & (listy[line - 1] == second)):
                denominator = denominator + 1
                if (listy[line] == third):
                    numerator = numerator + 1
    #                    print(line)
    #            print(check,denominator)

    if (denominator != 0):
        return numerator / denominator
    else:
        return 0


def secondtransitionlist(listy, rangexy):
    dictionary = {}
    rangey = rangexy.copy()
    rangey.append("")

    for i in range(len(rangey)):
        seconddictionary = {}
        for j in range(len(rangey)):
            thirddictionary = {}
            for k in range(len(rangey)):
                if ((i + j + k != 3 * len(rangey))):
                    thirddictionary[rangey[k]] = secondtransition(listy, rangey[i], rangey[j], rangey[k])
            seconddictionary[rangey[j]] = thirddictionary
        dictionary[rangey[i]] = seconddictionary
    return dictionary


# k=unseennumber(test,train)
# print(k)
# print(tag(train))
# print(transitionlist(handle(train),ytag(train)))

# viterbi(test,train,result,1)

# emissionx,emissiony=settle(train)
# print(protoemission(emissiony,emissionx,1))


def secondviterbi(testname, trainname, resultname, k):
    #    Initialization
    try:
        resultfile = open(resultname, "w+", encoding="UTF-8")
        current = []
        prob = []
        emissionx, emissiony = settle(trainname)
        edic, ydic = protoemission(emissiony, emissionx)
        x = extractx(testname)
        rangey = ytag(trainname)
        ydictionary = secondtransitionlist(handle(trainname), rangey)
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
            # if first start of sentence
            if (len(prob) == 0):
                for y in rangey:
                    probability = ydictionary[""][""][y]
                    word = node(probability, 7)
                    newlist.append(word)
                prob.append(newlist)
            # if second start of sentence
            elif (len(prob) == 1):
                latestprob = prob[len(prob) - 1]
                for i in range(len(rangey)):
                    argmax = 0
                    for j in range(len(rangey)):
                        initialprob = latestprob[j].prob
                        transitionprob = ydictionary[""][rangey[j]][rangey[i]]
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

                # if not start of sentence or end of sentence
            elif (wordie != ""):
                latestprob = prob[len(prob) - 1]
                for i in range(len(rangey)):
                    argmax = 0
                    for j in range(len(rangey)):
                        initialprob = latestprob[j].prob
                        transitionprob = ydictionary[rangey[latestprob[j].parent]][rangey[j]][rangey[i]]
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
                    transitionprob = ydictionary[rangey[prob[len(prob) - 1][j].parent]][rangey[j]][""]
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



# Read entities from predcition
def get_predicted(predicted, answers=defaultdict(lambda: defaultdict(defaultdict))):
    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_sent = ""
    last_entity = []

    answers[example] = []
    for line in predicted:
        line = line.strip()
        if line.startswith("##"):
            continue
        elif len(line) == 0:
            if entity:
                answers[example].append(list(entity))
                entity = []

            example += 1
            answers[example] = []
            word_index = 0
            last_ne = "O"
            continue
        else:
            split_line = line.split(separator)
            # word = split_line[0]
            value = split_line[outputColumnIndex]
            ne = value[0]
            sent = value[2:]

            last_entity = []

            # check if it is start of entity
            if ne == 'B' or (ne == 'I' and last_ne == 'O') or (last_ne != 'O' and ne == 'I' and last_sent != sent):
                if entity:
                    last_entity = list(entity)

                entity = [sent]

                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity = list(entity)
                entity = []

            if last_entity:
                answers[example].append(list(last_entity))
                last_entity = []

        last_sent = sent
        last_ne = ne
        word_index += 1

    if entity:
        answers[example].append(list(entity))

    return answers


# Read entities from gold data
def get_observed(observed):
    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_sent = ""
    last_entity = []

    observations = defaultdict(defaultdict)
    observations[example] = []

    for line in observed:
        line = line.strip()
        if line.startswith("##"):
            continue
        elif len(line) == 0:
            if entity:
                observations[example].append(list(entity))
                entity = []

            example += 1
            observations[example] = []
            word_index = 0
            last_ne = "O"
            continue

        else:
            split_line = line.split(separator)
            word = split_line[0]
            value = split_line[outputColumnIndex]
            ne = value[0]
            sent = value[2:]

            last_entity = []

            # check if it is start of entity, suppose there is no weird case in gold data
            if ne == 'B' or (ne == 'I' and last_ne == 'O') or (last_ne != 'O' and ne == 'I' and last_sent != sent):
                if entity:
                    last_entity = entity

                entity = [sent]

                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity = entity
                entity = []

            if last_entity:
                observations[example].append(list(last_entity))
                last_entity = []

        last_ne = ne
        last_sent = sent
        word_index += 1

    if entity:
        observations[example].append(list(entity))

    return observations


# Print Results and deal with division by 0
def printResult(evalTarget, num_correct, prec, rec):
    if abs(prec + rec) < 1e-6:
        f = 0
    else:
        f = 2 * prec * rec / (prec + rec)
    print('#Correct', evalTarget, ':', num_correct)
    print(evalTarget, ' precision: %.4f' % (prec))
    print(evalTarget, ' recall: %.4f' % (rec))
    print(evalTarget, ' F: %.4f' % (f))


# Compare results bewteen gold data and prediction data
def compare_observed_to_predicted(observed, predicted):
    correct_sentiment = 0
    correct_entity = 0

    total_observed = 0.0
    total_predicted = 0.0

    # For each Instance Index example (example = 0,1,2,3.....)
    for example in observed:

        if example in discardInstance:
            continue

        observed_instance = observed[example]
        predicted_instance = predicted[example]

        # Count number of entities in gold data
        total_observed += len(observed_instance)
        # Count number of entities in prediction data
        total_predicted += len(predicted_instance)

        # For each entity in prediction
        for span in predicted_instance:
            span_begin = span[1]
            span_length = len(span) - 1
            span_ne = (span_begin, span_length)
            span_sent = span[0]

            # For each entity in gold data
            for observed_span in observed_instance:
                begin = observed_span[1]
                length = len(observed_span) - 1
                ne = (begin, length)
                sent = observed_span[0]

                # Entity matched
                if span_ne == ne:
                    correct_entity += 1

                    # Entity & Sentiment both are matched
                    if span_sent == sent:
                        correct_sentiment += 1

    print()
    print('#Entity in gold data: %d' % (total_observed))
    print('#Entity in prediction: %d' % (total_predicted))
    print()

    prec = correct_entity / total_predicted
    rec = correct_entity / total_observed
    printResult('Entity', correct_entity, prec, rec)
    print()

    prec = correct_sentiment / total_predicted
    rec = correct_sentiment / total_observed
    printResult('Entity Type', correct_sentiment, prec, rec)


parser = argparse.ArgumentParser(description='Provide the training file, the test file, '
                                             'and the output file you wish to use.')
parser.add_argument('-train', type=str)
parser.add_argument('-test', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-standard', type=str)

args = parser.parse_args()
secondviterbi(args.test, args.train, args.output, 1)

separator = ' '
outputColumnIndex = 1
discardInstance = []
gold = open(args.standard, "r", encoding='UTF-8')
prediction = open(args.output, "r", encoding='UTF-8')
observed = get_observed(gold)
predicted = get_predicted(prediction)
compare_observed_to_predicted(observed, predicted)
