import numpy as np
import trains as tr
import time

#Creates initial population for genetic algorithm
def initialPopulation(stations,individuals,numTrainTour, numTrainRetour):
    meetingTrains = np.random.randint(2, size=(individuals, numTrainTour, numTrainRetour))
    passByTour = np.random.randint(2, size=(individuals, max(numTrainTour, numTrainRetour), max(numTrainTour,numTrainRetour)))
    return meetingTrains, passByTour

#Calculates list of fitness for all individuals in population
def getFitness(individuals, stations, time, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,frequencyOne, frequencyTwo, trainTime,meetingTrains, passByTour):
    fitnessList=np.zeros(individuals)
    for i in range(individuals):
        fitnessList[i]=tr.fitness(stations, time, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,frequencyOne, frequencyTwo, trainTime, passByTour[i], meetingTrains[i])
    return fitnessList

#This function handles the replacement, and calls for selection and mutation process
def replacement(individuals, meetingTrains, passByTour, fitnessList, maxRank, numTrainsTour, numTrainsRetour, replaceNumber, type, probability):
    if type=="rank":
        childrenMeeting, childrenPass=selection_rank(individuals, meetingTrains, passByTour, fitnessList, maxRank, numTrainsTour, numTrainsRetour, replaceNumber)
    elif type == "proportional":
        childrenMeeting, childrenPass = selection_proporcional(individuals, meetingTrains, passByTour, fitnessList, replaceNumber, numTrainsTour, numTrainsRetour)

    childrenMeeting, childrenPass = mutation(individuals, childrenMeeting, childrenPass, probability, numTrainsTour, numTrainsRetour)

    #This chunck replaces the R poorest individuals with children
    sortIndex=np.argsort(fitnessList)
    for i in range(replaceNumber):
        meetingTrains[sortIndex[individuals-1-i]]=childrenMeeting[i]
        passByTour[sortIndex[individuals - 1 - i]] = childrenPass[i]

    return meetingTrains, passByTour

#Calculates probabilities and finds two indiciduals to cross
def selection_rank(individuals, meetingTrains, passByTour, fitnessList, maxRank, numTrainsTour, numTrainsRetour, replaceNumber):
    childrenMeeting = np.zeros(((individuals, numTrainsTour, numTrainsRetour)))
    childrenPass = np.zeros(((individuals, max(numTrainsTour, numTrainsRetour), max(numTrainsTour, numTrainsRetour))))

    #By max/min formula
    min=2-maxRank

    #This chunck caculates rank for each individual
    sortIndex=np.argsort(fitnessList)
    otherSort=np.zeros(individuals)
    for i in range(individuals):
        otherSort[sortIndex[i]]=i+1

    #Calculates probabilities
    h=maxRank-(maxRank-min)*(otherSort-1)/(individuals-1)
    probabilities = h / individuals
    cumulative = np.cumsum(probabilities)

    #Chooses two individuals for crossing
    for i in range(replaceNumber):
        randomNumber=np.random.uniform()
        indexCross1=0
        for j in range(individuals):
            if randomNumber<=cumulative[j]:
                indexCross1=j
                break
        randomNumber = np.random.uniform()
        indexCross2 = 0
        for j in range(individuals):
            if randomNumber <= cumulative[j]:
                indexCross2 = j
                break


        childrenMeeting[i]=cross(meetingTrains[indexCross1], meetingTrains[indexCross2])
        childrenPass[i] = cross(passByTour[indexCross1], passByTour[indexCross2])

    return childrenMeeting, childrenPass

def selection_proporcional(individuals, meetingTrains, passByTour, fitnessList, replaceNumber, numTrainsTour, numTrainsRetour):
    childrenMeeting = np.zeros(((individuals, numTrainsTour, numTrainsRetour)))
    childrenPass = np.zeros(((individuals, numTrainsTour, numTrainsTour)))

    #Since fitness is about minimizing, the values for use in proportional selection is defined as population (max fitness - indidividual + 1)
    #+1 is because if all solutions are equal, we will get divide by zero
    fitnessNew=(max(fitnessList)+1)-fitnessList
    probabilities=fitnessNew/(sum(fitnessNew))
    cumulative=np.cumsum(probabilities)
    for i in range(replaceNumber):
        randomNumber=np.random.uniform()
        indexCross1=0
        for j in range(individuals):
            if randomNumber<=cumulative[j]:
                indexCross1=j
                break
        randomNumber = np.random.uniform()
        indexCross2 = 0
        for j in range(individuals):
            if randomNumber <= cumulative[j]:
                indexCross2 = j
                break

        childrenMeeting[i] = cross(meetingTrains[indexCross1], meetingTrains[indexCross2])
        childrenPass[i] = cross(passByTour[indexCross1], passByTour[indexCross2])

    return meetingTrains, passByTour

#Crosses two individuals, randomly choosing each bit from one of the parents, returning a child
def cross(matrixOne, matrixTwo):
    rows=len(matrixOne)
    columns=len(matrixOne[0])
    newMatrix1=np.zeros((rows, columns))
    newMatrix2=np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if np.random.uniform()<=0.5:
                newMatrix1[i][j] = matrixOne[i][j]
            else:
                newMatrix1[i][j] = matrixTwo[i][j]
    return newMatrix1

#iterates through lists and changes with a probability
#This should be implemented more effectively
def mutation(individuals, meetingTrains, passByTour, probability, numTrainsTour, numTrainsRetour):
    for i in range(individuals):
        for j in range(numTrainsTour):
            for k in range(numTrainsRetour):
                if np.random.uniform()<probability:
                    meetingTrains[i][j][k]=np.random.randint(2)

    for i in range(individuals):
        for j in range(max(numTrainsTour, numTrainsRetour)):
            for k in range(max(numTrainsTour, numTrainsRetour)):
                if np.random.uniform()<probability:
                    passByTour[i][j][k]=np.random.randint(2)

    return meetingTrains, passByTour

#This function runs the genetic algorithm for a specified amount of time, and returns a convergence list based on computation timetime
def convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains,passByTour,probability, numTrainTour, numTrainRetour, maxRank, replaceNumber, type):
    bestList=np.zeros(runTime//5) #every 5 seconds, new fitness into convergence plot
    i=0
    threshold=5

    fitnessList = getFitness(individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour)

    start = time.time()
    now= time.time()
    #Keeps going until we reach specified time, performs one generation of evolution each round
    while (now-start)<runTime:
        meetingTrains, passByTour = replacement(individuals, meetingTrains, passByTour, fitnessList, maxRank, numTrainTour, numTrainRetour, replaceNumber, type, probability)
        fitnessList = getFitness(individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour)
        #Add to convergence list every given time
        if now-start>i*threshold:
            best = np.min(fitnessList)
            bestList[i]=best
            print(best)
            i+=1

        now=time.time()
    x=np.linspace(0,runTime,runTime//threshold)
    return x, bestList

#This is ecual to the one above, but customized to the last part of main, returning meetingTrains and passByTour
def convergence2(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains,passByTour,probability, numTrainTour, numTrainRetour, maxRank, replaceNumber, type):
    bestList=np.zeros(runTime//1)
    i=0
    threshold=1

    fitnessList = getFitness(individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour)

    start = time.time()
    now= time.time()
    while (now-start)<runTime:
        meetingTrains, passByTour = replacement(individuals, meetingTrains, passByTour, fitnessList, maxRank, numTrainTour, numTrainRetour, replaceNumber, type, probability)
        fitnessList = getFitness(individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour)
        print(np.min(fitnessList))
        if now-start>i*threshold:
            best = np.min(fitnessList)
            bestList[i]=best
            i+=1
            bestIndex=np.argmin(fitnessList)
        now=time.time()
    x=np.linspace(0,runTime,runTime//threshold)
    return x, bestList, meetingTrains[bestIndex], passByTour[bestIndex]