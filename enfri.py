import numpy as np
import trains as tr
import time

#performs one step of simulated annealing
def enfri(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains, T, numTrainTour, numTrainRetour, oldFitness):
    case=0
    #Choose random neighbour
    if np.random.randint(2):
        a=np.random.randint(numTrainTour)
        b=np.random.randint(numTrainRetour)
        if meetingTrains[a][b]==0:
            meetingTrains[a][b]=1
            case=1
        else:
            meetingTrains[a][b]=0
            case=2

    else:
        a=np.random.randint(max(numTrainTour, numTrainRetour))
        b=np.random.randint(max(numTrainTour, numTrainRetour))
        if passByTour[a][b]==0:
            passByTour[a][b]=1
            case=3
        else:
            passByTour[a][b]=0
            case=4

    #Check fitness
    newFitness=tr.fitness(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains)
    #Decide whether to choose new step
    if newFitness>oldFitness:
        probability=np.exp((oldFitness-newFitness)/T)
        if np.random.uniform()>probability:
            #It is now rejected
            newFitness=oldFitness
            if case==1:
                meetingTrains[a][b] = 0
            elif case==2:
                meetingTrains[a][b] = 1
            elif case==3:
                passByTour[a][b] = 0
            elif case==4:
                passByTour[a][b] = 1
    return meetingTrains, passByTour, newFitness

def alpha1(T, k):
    return k*T

def alpha2(T, k):
    return T/(1+k*T)

#Similar to convergence in genetic, runs each "generation" of simulated annealing
def run2(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, T, numTrainTour, numTrainRetour, k, type):
    threshold = 1
    bestList = np.zeros(runTime // 1)
    i = 0

    passByTour = np.random.randint(2, size=(max(numTrainTour, numTrainRetour), max(numTrainTour, numTrainRetour)))
    meetingTrains = np.random.randint(2, size=(numTrainTour, numTrainRetour))

    fitness = tr.fitness(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains)

    start = time.time()
    now = time.time()
    while (now - start) < runTime:
        meetingTrains, passByTour, fitness = enfri(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains, T, numTrainTour, numTrainRetour, fitness)
        print(fitness)
        if type==1:
            T=alpha1(T, k)
        elif type==2:
            T=alpha2(T, k)
        if now - start > i * threshold:
            bestList[i] = fitness
            i += 1
        now = time.time()
    x = np.linspace(0, runTime, runTime // threshold)
    return x, bestList, meetingTrains, passByTour

#As run1, but not adapted to last part of main
def run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, T, numTrainTour, numTrainRetour, k, type):
    bestList = np.zeros(runTime // 5)
    i = 0
    threshold = 5
    passByTour = np.random.randint(2, size=(numTrainTour, numTrainTour))
    meetingTrains = np.random.randint(2, size=(numTrainTour, numTrainRetour))

    fitness = tr.fitness(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains)

    start = time.time()
    now = time.time()
    while (now - start) < runTime:
        meetingTrains, passByTour, fitness = enfri(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains, T, numTrainTour, numTrainRetour, fitness)
        if type==1:
            T=alpha1(T, k)
        elif type==2:
            T=alpha2(T, k)
        if now - start > i * threshold:
            bestList[i] = fitness
            i += 1
        now = time.time()
    x = np.linspace(0, runTime, runTime // threshold)
    return x, bestList