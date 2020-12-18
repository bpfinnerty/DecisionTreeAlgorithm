#
# Author: Brian Finnerty
# file: AdaBoost.py
# This program will utilize Decision Trees or Adaboost with decision stumps
# to distinguish between the language of dutch and english given 15 words
#
# To run this file please first specify if you would like to train, or predict
# train <trainingFile> <writeObjectFile> <TypeOfLearningAlgo>
#
# please specify the training file with examples and answers in the form of nl|
# or en|
#
# Then specify where the object should be written and what algorithm you want
# The program will output a model of DT or Ada and serialize it to the output File
#
# predict <learnedObject> <testingData>
#
# please specify the model of trained algorithm you would like to test and
# then data to test it on
#
# this program will then display the answers it came up with to the testing set
#

import sys
import re
import math
import pickle

# index of availible slot in my array for the DT
nextAvailible = 2

"""
    :arg
        inFile: the input file of all the the 15 line words I must parse
    This function will take strings of words and form a list of boolean 
    vectors that will allow me to get a better understanding of how each
    sentence fits into my chosen features
    :return A list of lists that hold boolean values
"""
def makeBoolArray(inFile):
    i = 0
    stringArray= [0]
    # tests strings on all attributes I came up with
    with open(inFile, "r", encoding="utf-8") as f:
        for line in f:
            shortCounter = 0
            longCounter = 0
            stringArray.append([0])
            stringArray[i] = [False] * 33
            line = line.strip()
            parsed = re.split("[ ;,.|()\s]\s*",line)
            line = line.lower()
            if line.count("j") >=3:
                stringArray[i][1] = True
            if line.count("z") >=2:
                stringArray[i][2] = True
            if "et" in line:
                stringArray[i][3] = True
            if "aan" in line:
                stringArray[i][4] = True
            if "ij" in line:
                stringArray[i][5] = True
            if "aar" in line:
                stringArray[i][6] = True
            if "ë" in line:
                stringArray[i][7] = True
            if "á" in line:
                stringArray[i][8] = True
            if "í" in line:
                stringArray[i][9] = True
            if "ó" in line:
                stringArray[i][10] = True
            if "ú" in line:
                stringArray[i][11] = True
            if "ö" in line:
                stringArray[i][12] = True
            if "ü" in line:
                stringArray[i][13] = True
            if "é" in line:
                stringArray[i][14] = True

            if parsed[0] == "en":
                stringArray[i][0] = "en"
            elif parsed[0] == "nl":
                stringArray[i][0] = "nl"

            for word in parsed:
                word = word.lower()
                if "for" == word:
                    stringArray[i][15] = True
                if word == "het":
                    stringArray[i][16] = True
                if word == "it":
                    stringArray[i][17] = True
                if word == "als":
                    stringArray[i][18] = True
                if word == "and":
                    stringArray[i][19] = True
                if word == "zijn":
                    stringArray[i][20] = True
                if word == "be":
                    stringArray[i][21] = True
                if word == "the":
                    stringArray[i][22] = True
                if word == "to":
                    stringArray[i][23] = True
                if word == "een":
                    stringArray[i][24] = True
                if word == "dat":
                    stringArray[i][25] = True
                if word == "voor":
                    stringArray[i][26] = True
                if word == "have":
                    stringArray[i][27] = True
                if word == "van":
                    stringArray[i][28] = True
                if len(word) == 1:
                    shortCounter = shortCounter+1
                if len(word) >= 11:
                    longCounter = longCounter+1
                if word == "are":
                    stringArray[i][31]= True
                if word == "de":
                    stringArray[i][32] = True
            if shortCounter>=2:
                stringArray[i][29] = True
            if longCounter >=3:
                stringArray[i][30] = True
            i = i+1
        x = stringArray.pop()
        return stringArray

"""
    :arg
        dtree: an object that has to be serialized
        outPut: the file to serialize the data to
    This function will serialize the object I send to it to the output file
"""
def writeToFile(dTree,output):
    with open(output, "wb") as f:
        pickle.dump(dTree,f)


"""
    :arg
        boolArray: the list of boolean vectors
        X: the attribute i want to split on
        b: the boolean value that I am looking for
    This function will create a list that contains all of the vectors that 
    fulfill the boolean value b at position X. So if i wanted all vectors
    that had trues for space three, then that is the list of vectors
    that I would get.
    :return
        a list of boolean vectors
        
"""
def getArray(boolArray,X,b):
    array = [0]
    i = 0
    for elem in boolArray:
        if elem[X]==b:
            array.append([0])
            array[i]=elem
            i=i+1
    array.pop()
    return array

"""
    calcEntropy
    :arg
        boolArray: Array of boolean vectors
        X: The attribute we are splitting on
        
    This function will divide the boolArray into true and false based on the
     splitting attribute
    It will then sum up all english trues, dutch trues, english falses and 
    dutch falses.
    Then it will calculate the entropy of both sides and sum them, thus 
    providing the total entropy.
    :return a float of the entropy
"""
def calcEntropy(boolArray,X):
    # counters to make my life easier
    totalCount = len(boolArray); tCount = 0; fCount = 0
    eTrueCount=  0; dTrueCount = 0; eFalseCount = 0; dFalseCount = 0
    # steps trhough all elements and counts up the number of occurances of
    # true's and falses for both en and dutch
    for elem in boolArray:
        if elem[X]:
            tCount = tCount+1
            if elem[0] == "en": eTrueCount = eTrueCount + 1
            else: dTrueCount = dTrueCount + 1
        else:
            fCount = fCount+1
            if elem[0] == "en": eFalseCount = eFalseCount + 1
            else: dFalseCount = dFalseCount + 1

    # equation: simplified one that appeared on the video of log_2(1/p_k))
    # Now I want to calculate the entropy on the true side
    sumTrue = (eTrueCount+dTrueCount); entroTrue = 0
    trueBVal = 0; falseBVal = 0
    if sumTrue != 0:
        eTruePart = eTrueCount/sumTrue; dTruePart = (dTrueCount/sumTrue)
        ePart = 0; dPart = 0
        # ensure division by zero doesn't happen
        if eTruePart !=0:
            ePart = eTruePart * math.log((1 / eTruePart), 2)
        if dTruePart !=0:
            dPart = dTruePart*math.log((1/dTruePart),2)

        entroTrue = (sumTrue/totalCount) * (ePart + dPart)
        trueBVal = (sumTrue/totalCount)* math.log((sumTrue/totalCount),2)

    # same equation for the other side now
    # false block entropy
    sumFalse = eFalseCount + dFalseCount;  entroFalse = 0
    if sumFalse != 0:
        eFalsePart = eFalseCount / sumFalse
        dFalsePart = dFalseCount / sumFalse
        ePart = 0; dPart = 0
        # prevent division by zero
        if eFalsePart != 0:
            ePart = eFalsePart * math.log((1 / eFalsePart), 2)
        if dFalsePart != 0:
            dPart = dFalsePart * math.log((1 / dFalsePart), 2)

        entroFalse = (sumFalse / totalCount) * (ePart + dPart)
        falseBVal = (sumFalse/totalCount)* math.log((sumFalse/totalCount),2)
    # the entropy of both true and false will be the total entropy of
    # this attribute
    remainder = entroTrue+entroFalse
    bVal = (trueBVal+falseBVal)*-1
    return bVal-remainder


"""
    :arg
        boolArray: an array of Boolean vectors
    This function will determine if all remaining elements in boolArray 
    are of the same langauge.
    If true, it will return true, it false it will return false.
    :return boolean value
"""
def allSame(boolArray):
    # loops through the list of vectors
    if len(boolArray) > 0:
        langStr = boolArray[0]
        langStr = langStr[0]
        for elem in boolArray:
            if elem[0] != langStr:
                # if an element is varried then it will return false
                # since there is a difference
                return False
        # returns true if all are the same
        return True
    # no values to check
    return False

"""
    :arg
        boolArray: Array of boolean vectors
    This function will calculate the majority language in the bool 
    array and return it
    :return string that corresponds to majority language
"""
def getMajority(boolArray):
    enCount = 0
    nlCount = 0
    # count up dutch and english in the list
    for elem in boolArray:
        if elem[0] == "en":
            enCount = enCount+1
        else:
            nlCount = nlCount+1
    # return appropriate string for the majoirity
    if enCount>=nlCount:
        return "en"
    else:
        return "nl"

"""
    :arg
        decisionTree: Array of dictionaries that contain the important 
        info of that node
        spot: the current value of the node we are at
    This function will calculate the parent of spot and then return the 
    value associated with it
    :return
        A string corresponding to the majority language of the parent node    
"""
def getParent(decisionTree,spot):
    string = ""
    if spot == 0:
        sys.stderr.write("decisionTree has no Data")
        exit(1)
    else:
        dict = decisionTree[spot]
        string = dict["majority"]
    return string


"""
    :arg
        decisionTree: A list of dictionaries that will act as nodes
        boolArray: a list of boolean vectors that will be used for training
        attributeSet: the availible values still left to be used
        spot: the position that the current node is at in the dt list
        parent: the parent of this node
    This function will calculate the importance of an attribute, pick the one
    with the most importance, then split the data along that factor anc
    contunie recursively building nodes until one of three conditions are met:
        the boolArray have all the same language types
        There are no more attributes to look at
        the boolArray is empty.
    I will then return the majority or parent node's answer based on what I 
    have
    :return
        A list of dictionaries that represent my DT
"""
def decisionLearning(decisionTree,boolArray,attributeSet, spot, parent):
    # forms leaf in tree if all values are the same
    if allSame(boolArray):
        answer = getMajority(boolArray)
        dict = {
            "type": "leaf",
            "majority": answer,
            "parent": parent
            }
        decisionTree[spot] = dict
        return decisionTree
    # ran out of attributes to parse on
    elif len(attributeSet) == 0:
        answer = getMajority(boolArray)
        dict = {
            "type": "leaf",
            "majority": answer,
            "parent": parent
        }
        decisionTree[spot] = dict
        return decisionTree
    # ran out of elements to look at
    elif len(boolArray) == 0:
        answer = getParent(decisionTree, parent)
        dict = {
            "type": "leaf",
            "majority": answer,
            "parent": parent
            }
        decisionTree[spot] = dict
        return decisionTree
    # we still recurse
    else:
        # list of remainders for each set
        remainder = [0]*33
        for elem in attributeSet:
            gain = calcEntropy(boolArray, elem)
            remainder[elem] = gain
        # the most information gained will correspond to the attribute
        # with the smallest remainder
        maxGainPos = attributeSet[0]
        maxGain = remainder[maxGainPos]
        # this will get the most info gained by splitting
        for elem in attributeSet:
            gain = remainder[elem]
            if gain > maxGain:
                maxGain = gain
                maxGainPos = elem
        # arrays that split down my data
        tArray = getArray(boolArray,maxGainPos,True)
        fArray = getArray(boolArray,maxGainPos,False)
        # copy over attributes for each branch
        attributeSet.remove(maxGainPos)
        aTrue = attributeSet.copy()
        aFalse = attributeSet.copy()

        # claculate where the children of this node will be
        global nextAvailible
        lSpot = nextAvailible
        rSpot = nextAvailible+1
        nextAvailible = nextAvailible+2
        dict = {
                   "type": "node",
                   "attribute": maxGainPos,
                   "majority": getMajority(boolArray),
                   "trueChild": lSpot,
                   "falseChild": rSpot,
                   "parent": parent}
        # append more space if required
        if lSpot >= len(decisionTree) or rSpot >= len(decisionTree):
            i = 0
            for i in range(20):
                decisionTree.append({"type":"none"})
        decisionTree[spot] = dict
        # recurse on both the true and false branches until a leaf is reached
        decisionTree = decisionLearning(decisionTree,tArray,aTrue,lSpot, spot)
        decisionTree = decisionLearning(decisionTree,fArray,aFalse,rSpot, spot)

        return decisionTree

"""
    :arg
        boolList: a list of boolean vectors
        weightList: a list of weights for each example
        attribute: the attribute I will be looking at
    This will calculate the importance of an attribute based on its remainder
    It will recieve the list of vectors and sum up the appropriate weights for
    the true and false subsets on that attribute. Then it will use the
    remainder function that was used earlier, but it will now utilize weights
    rather than total counts
    :return 
        the remainder of the attribute
"""
def calcImportance(boolList, weightList, attribute):
    # counters
    trueVal = 0;falseVal = 0;tE = 0;tD = 0;fE = 0;fD = 0
    counter = 0
    # loop through to get all the diff weights
    for line in boolList:
        if line[attribute]:
            trueVal = trueVal+weightList[counter]
            if line[0] == "en":
                tE = tE+weightList[counter]
            else:
                tD = tD+ weightList[counter]
        else:
            falseVal = falseVal+weightList[counter]
            if line[0] == "en":
                fE = fE + weightList[counter]
            else:
                fD = fD + weightList[counter]
        counter = counter+1
    # this will preform the math
    remainder = 0
    trueEnglish = 0; trueDutch=0; falseEnglish = 0; falseDutch = 0
    # true node calculations
    if trueVal!=0:
        if tE != 0:
            trueEnglish = (tE/trueVal) * math.log((1/(tE/trueVal)),2)
        if tD !=0:
            trueDutch = (tD/trueVal) * math.log((1/(tD/trueVal)),2)
    # false side calculations

    if falseVal!=0:
        if fE != 0:
            falseEnglish = (fE/falseVal) * math.log((1/(fE/falseVal)),2)
        if fD !=0:
            falseDutch = (fD/falseVal) * math.log((1/(fD/falseVal)),2)

    # the remainder function
    remainder = ((trueVal/1) *(trueEnglish+trueDutch)) + \
                 ((falseVal/1) * (falseEnglish+falseDutch))
    b = 0
    if trueVal != 0 and falseVal !=0:
        b = (trueVal* math.log(trueVal,2) + falseVal*math.log(falseVal,2))*-1

    importance = b - remainder
    return importance

"""
    :arg
        boolList: list of boolean Vectors
        attribtue: the specific attribute I will be splitting on
        b: boolean value
    Get the majority of a specific attribute for a specific boolean
    :return
        the string associated to the majority
"""
def getAttributeMajority(boolList,attribute,b,weightList):
    enCount = 0
    nlCount = 0
    counter = 0
    for elem in boolList:
        if elem[attribute]==b:
            if elem[0]=="en":
                enCount = enCount+weightList[counter]
            else:
                nlCount=nlCount+weightList[counter]
        counter = counter+1
    if enCount > nlCount:
        return "en"
    else:
        return "nl"

"""
    :arg
        boolList: List of boolean Vectors
        weightLIst: list of the weights correcponding to the bList
        attribute: the attribute I will be splitting on
        trueMajority: the string that reps the majority of true values
        falseMajojity: represents the lang of false values
    This function will calculate the error for a specific attribute
    by looking at all the incorectly identified values
    :return
        sum of all the errors
"""
def calcError(boolList,weightList,attribute, trueMajority,falseMajority):

    counter = 0
    err = 0
    for elem in boolList:
        if elem[attribute] == True and  elem[0]!=trueMajority:
            err = err + weightList[counter]
        elif elem[attribute]== False and elem[0]!=falseMajority:
            err = err +weightList[counter]
        counter= counter+1
        # sums up all the missmatched value's errors
    return err

"""
    :arg
        boolList: List of boolean Vectors
        weightLIst: list of the weights correcponding to the bList
        update: the value to multiply all correct values by to decrease
        their importance in the next iteration
        
        attribute: the attribute I will be splitting on
        trueMajority: the string that reps the majority of true values
        falseMajojity: represents the lang of false values
        
    This function will change all correct values to boost theweight of the
    incorrect values so then we may be able to learn the language next time
    :return
        list of newly weighted values
"""
def updateWeightList(boolList,weightList,update,attribute,trueMajority,falseMajority):

    i = 0
    for bVector in boolList:
        if bVector[attribute]:
            if bVector[0]==trueMajority:
                weightList[i] = weightList[i] * update
        else:
            if bVector[0] == falseMajority:
                weightList[i] = weightList[i] * update
        i = i+1

    normalizeDivider = 0
    for err in weightList:
        normalizeDivider = normalizeDivider + err

    for i in range(len(weightList)):
        weightList[i] = weightList[i]/normalizeDivider
    return weightList


"""
    :arg
        boolList: List of boolean Vectors
        weightLIst: list of the weights correcponding to the bList
        maxStump: the K value that limits the num of hypothesis
        attributeSet: the list of attributes to split my data by
        stumpList: a list of dicts keeping track of all other hypothesis
    This function will iteratively create a stump, figure out its error,
    update the weights for all correct answers and then repeate this 
    proccess until we have k hypothesis
    :return
        fleshed out list of hypothesis'
"""
def makeAda(boolArray,weightList,maxStump,attributeSet,stumpList):
    k = 0
    # loop for k stumps
    for k in range(maxStump):
        importanceList = [0]*33
        importance = -10
        importPos = -1
        # importance should be related to the least remainder
        for i in attributeSet:
            imp = calcImportance(boolArray,weightList,i)
            importanceList[i] = imp
            if imp > importance:
                importance = imp
                importPos = i
        # get the majorities for this attribute on true and on false
        trueMajority = getAttributeMajority(boolArray,importPos,True,weightList)
        falseMajority = getAttributeMajority(boolArray,importPos,False,weightList)
        error = calcError(boolArray,weightList,importPos, trueMajority
                          ,falseMajority)
        #if error>=.49:
        #    return stumpList
        # this is the value to update all correct answers by
        update = error / (1 - error)
        # changes the weighted list
        weightList = updateWeightList(boolArray,weightList,update,
                                      importPos,trueMajority,falseMajority)
        tot = 0
        # the weight to trust the hypothesis
        finalWeight = math.log(((1-error)/error))
        # -1 = dutch, 1 = english
        putT = -1
        putF = -1
        if trueMajority == "en":
            putT = 1
        if falseMajority == "en":
            putF = 1
        # form a hypothesis for all of this
        dict = {
            "attribute": importPos,
            "weight": finalWeight,
            "trueBranch": putT,
            "falseBranch": putF
        }
        # add it to the list and continue
        stumpList.append(dict)
        k = k+1

    return stumpList

"""
    The main function that will deal with all the set uo
    and finally the display of the answers
"""
def main():
    # get arguments
    whatToDo = sys.argv[1]
    file1 = sys.argv[2]
    file2 = sys.argv[3]
    if len(sys.argv) == 5:
        withWhat = sys.argv[4]
    # this is the training section
    if whatToDo == "train":
        # gets list of boolean Vectors
        boolArray = makeBoolArray(file1)
        # all possible attributes to test
        attributeSet = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
            ,20,21,22,23,24,25,26,27,28,29,30,31,32]
        # this will make a decision tree
        if withWhat == "dt":
            decisionTree = [{"algo":"dt"}] * 50
            decisionLearning(decisionTree,boolArray,attributeSet, 1,1)
            writeToFile(decisionTree, file2)
        # this will make the adaBoost algorithm
        else:
            # figure out starting weight for everyone
            originalWeight = (1/len(boolArray))
            weightList = [originalWeight] * (len(boolArray))
            stumpList = [{"algo":"ada"}]
            # number of hypothesis
            maxStump = 10
            stumpList = makeAda(boolArray,weightList,maxStump,attributeSet,stumpList)
            writeToFile(stumpList,file2)
    else:
        # gets data from the serialized file
        dictArray = []
        with open(file1, "rb") as f:
            dictArray = pickle.load(f)
        testArray = makeBoolArray(file2)
        algo = dictArray[0]
        # loop through all test examples
        for line in testArray:
            i = 1
            go = True
            # data tree serialized
            if algo["algo"]=="dt":
                while go:
                    dict = dictArray[i]
                    # print out answer of dataTree
                    if dict["type"] == "leaf":
                        print(dict["majority"])
                        go = False
                    # decide which node to go to next in tree
                    elif dict["type"] == "node":
                        split = dict["attribute"]
                        if line[split]:
                            i = dict["trueChild"]
                        else:
                            i = dict["falseChild"]
                    else:
                        print("Reached Nothing. Fix the tree")
                        exit(1)
            # AdaBoost serialized
            else:
                j = 1
                go = True
                hypothesisSum = 0
                # find out all the hypothesis and sum up their answers
                while j<len(dictArray):
                    dict = dictArray[j]
                    at = dict["attribute"]
                    if line[at]:
                        hypothesisSum = hypothesisSum + dict["weight"]*dict["trueBranch"]
                    else:
                        hypothesisSum = hypothesisSum + dict["weight"]*dict["falseBranch"]
                    j=j+1
                # display answer
                if hypothesisSum<0:
                    print("nl")
                elif hypothesisSum>=0:
                    print("en")


if __name__ == "__main__":
    main()
