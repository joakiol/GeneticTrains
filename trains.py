from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import timeit

#Calculates how many trains will run from one side, and which types of trains they are
def calcNumTrain(startTimeOne, startTimeTwo, frequencyOne, frequencyTwo, time):
    number=0
    trainTypes=[]
    for i in range(time+1):
        if startTimeOne==0:
            number+=1
            startTimeOne+=frequencyOne
            trainTypes.append(1)
        if startTimeTwo== 0:
            number += 1
            startTimeTwo += frequencyTwo
            trainTypes.append(2)
        startTimeOne-=1
        startTimeTwo-=1

    return number, trainTypes

#Takes as argument a prepared list of the trains movement, to be plottet
def plotTrains(trainsTourPlot, time, numTrainTour, stations, typeTour, trainTime, trainTypes):

    y = np.linspace(1, time, time)

    #plots the stations at their respective position on the y-axis
    stationPlots=np.zeros((stations, time))
    sumStation=0
    for i in range(0,stations-1):
        sumStation += trainTime[0][i]
        for j in range(time):
            stationPlots[i+1][j]=sumStation
        plt.plot(y, stationPlots[i], c='black', linewidth=0.5)

    #plots the movement of the trains
    for i in range(numTrainTour):

        start = False
        end = False

        #This loop will run though the plotting lists, and alter the part after the train has finished running, which has not been changed previously
        for j in range(time):

            #when a retour-train is finished, the remaining values should be zero
            if start and typeTour=='retour':
                trainsTourPlot[i][j]=0

            #when a tour-train is finished, the remaining values should be equal to the value of the end station
            if end and typeTour=='tour':
                trainsTourPlot[i][j]=sum(trainTime[0])

            #This boolean marks a train which has reached zero
            if -1e-5<trainsTourPlot[i][j] and trainsTourPlot[i][j]<1e-5:
                start=True

            #This boolean marks a train which has reached end station
            if sum(trainTime[0])-1e-5<trainsTourPlot[i][j] and trainsTourPlot[i][j]<sum(trainTime[0])+1e-5:
                end=True

        if trainTypes[i]==1:
            plt.plot(y, trainsTourPlot[i], c='green')
        else:
            plt.plot(y, trainsTourPlot[i], c='blue')

#This function runs all the trains, and calculates total train run time, and total train waiting time
#This function is rather big and messy, but is made with computation speed in mind. Have therefore reduced function calls.
def fitness(stations, time, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains):

    strechesTrain = [0] * (stations - 1)    #List to keep track of which stretches are busy
    trainsTour = deque()                    #List to keep track of all moving tour-trains
    trainsRetour = deque()                  #List to keep track of all moving retour-trains

    numberTour = 0                          #Keeps track of current tour/retour-train
    numberRetour = 0

    numberPopTour = 0                       #Will be incremented to remove finished trains at the end of one round of movement
    numberPopRetour = 0

    totalTime=0                             #Used to represent fitness
    waitingTime=0

    #Moves train one minute at a time
    for minutes in range(time):

        #This chunck starts new trains at their schedule, and adds them to trainsTour/trainsRetour
        if startTimeOneTour == 0:
            numberTour += 1
            startTimeOneTour += frequencyOne
            trainsTour.append([0, numberTour, 0, minutes])

        if startTimeOneRetour == 0:
            numberRetour += 1
            startTimeOneRetour += frequencyOne
            trainsRetour.append([0, numberRetour, stations - 1, minutes])

        if startTimeTwoTour == 0:
            numberTour += 1
            startTimeTwoTour += frequencyTwo
            trainsTour.append([1, numberTour, 0, minutes])

        if startTimeTwoRetour == 0:
            numberRetour += 1
            startTimeTwoRetour += frequencyTwo
            trainsRetour.append([1, numberRetour, stations - 1, minutes])

        startTimeOneTour -= 1
        startTimeOneRetour -= 1
        startTimeTwoTour -= 1
        startTimeTwoRetour -= 1

        #########################################################################

        #This chunck moves all trains in trainsTour
        for i in range(len(trainsTour)):
            stretch = int(trainsTour[i][2] + 1e-5)          #calculates the current stretch for movement (trainsTour[i][2] stores the position-information). The +1e-5 is to override possible rounding errors
            stretchMinus = int(trainsTour[i][2] - 1e-5)     #If the train is at a station, this will result in a different stretch
            trainsTour[i][2] += 1 / trainTime[trainsTour[i][0]][stretch]

            #This part investigates if trains have passed each other
            for l in range(i, 0, -1):

                #Checks if a train has passed another train
                if trainsTour[i][2] > trainsTour[l - 1][2] + 1e-5:

                    #Checks if they just left a station. In that case, passing is fine
                    if stretch == stretchMinus:

                        #passByTour contains information on whether the train is allowed to pass
                        if 1 == passByTour[trainsTour[i][1] - 1][trainsTour[l - 1][1] - 1]:

                            #In this case the train is allowed to pass. The other train is pushed back to the last station(where it can be passed), and then moved as far as it would have gotten after waiting.
                            trainsTour[l - 1][2] = stretch
                            numbersBackPassingTrain = (trainsTour[i][2] % 1) * trainTime[trainsTour[i][0]][stretch] #How many times the passing train has moved since last station
                            numbersBackWaitingTrain = (trainsTour[l-1][2] % 1) * trainTime[trainsTour[l-1][0]][stretch] #How many times the waiting train has moved since last station
                            waitingTime+=numbersBackWaitingTrain-numbersBackPassingTrain        #It is equal to the waiting time of the train which is being passed

                            #Finally the train moves as far as it would have gotten after waiting
                            for p in range(minutes - int(numbersBackPassingTrain + 1e-5) + 1, minutes + 1):
                                trainsTour[l - 1][2] += 1 / trainTime[trainsTour[l - 1][0]][stretch]

                        else:
                            #in this case the train is not allowed to pass. It must wait, and waiting time is incremented.
                            trainsTour[i][2] -= 1 / trainTime[trainsTour[i][0]][stretch]
                            waitingTime+=1
                            break               #It is not necessary to see if it must wait for other trains as well, thus we break.

            #If a train is finished, it must be popped afterwards
            if int(trainsTour[i][2] + 1e-5) == stations - 1:
                numberPopTour += 1

            #Marks the stretch as busy
            strechesTrain[stretch] = 1


##############################################################################

        #This chunck investigates if a retour-train have to wait for a tour-train, or not
        for i in range(len(trainsRetour)):

            stretch = int(trainsRetour[i][2] - 1e-5)
            cantMove = False

            #Checks if the stretch is busy
            if strechesTrain[stretch] == 1:

                #For all tour-trains:
                for j in range(len(trainsTour) - 1, -1, -1):

                    #If they are on the same stretch
                    if int(trainsTour[j][2] - 1e-5) == stretch:

                        #meetingTrains stores information on which train has to wait
                        if meetingTrains[trainsTour[j][1] - 1][trainsRetour[i][1] - 1] == 0:
                            cantMove = True

            if cantMove:
                #In this case the retour-train must wait, and all tour-trains are left as they are
                continue
            else:
                #In this case all the conflicting tour-trains will be pushed back.
                if strechesTrain[stretch] == 1:
                    strechesTrain[stretch] = 0 #The busy-label is removed, since the tour-trains are now waiting
                    for j in range(len(trainsTour) - 1, -1, -1):
                        if int(trainsTour[j][2] - 1e-5) == stretch:
                            if int(trainsTour[j][2] + 1e-5) == stations - 1: #The train might just have arrived at end station, and is ready to be popped. But if it is pushed back, it will not be popped.
                                numberPopTour -= 1
                            numbersBack=(trainsTour[j][2] % 1) * trainTime[trainsTour[j][0]][stretch]
                            waitingTime+=numbersBack
                            trainsTour[j][2] = stretch  #the train is moved to last station.


        ##############################################################################

        #This chunck moves the retour trains, if there is no conflict. (Any conflict where they would have "won", have been altered to a no-conflict-label).
        for i in range(len(trainsRetour)):
            stretch = int(trainsRetour[i][2] - 1e-5)
            stretchMinus = int(trainsRetour[i][2] + 1e-5)

            #Stretch is not busy - train can go
            if strechesTrain[stretch]==0:
                trainsRetour[i][2] -= 1 / trainTime[trainsRetour[i][0]][stretch]

                #This chunck is similar to the one on trainsTour, searching for passing trains.
                for l in range(i, 0, -1):
                    if trainsRetour[i][2] < trainsRetour[l - 1][2] - 1e-5:
                        if stretch == stretchMinus:
                            if 1 == passByTour[trainsRetour[l-1][1] - 1][trainsRetour[i][1] - 1]: #Notice the different indexing. This to access the upper right part of the matrix (trainsTour will access lower left part).
                                trainsRetour[l - 1][2] = stretch + 1
                                numbersBackPassingTrain = (stretch+1-trainsRetour[i][2]) * trainTime[trainsRetour[i][0]][stretch]
                                numbersBackWaitingTrain = (stretch+1-trainsRetour[l-1][2]) * trainTime[trainsRetour[l-1][0]][stretch]
                                waitingTime+=numbersBackWaitingTrain-numbersBackPassingTrain
                                for p in range(minutes - int(numbersBackPassingTrain + 1e-5) + 1, minutes + 1):
                                    trainsRetour[l - 1][2] -= 1 / trainTime[trainsRetour[l - 1][0]][stretch]

                            else:
                                trainsRetour[i][2] += 1 / trainTime[trainsRetour[i][0]][stretch]
                                waitingTime+=1
            else:
                #In this case the stretch is busy, meaning the train must be sendt back
                numbersBack = (stretch+1 - (trainsRetour[i][2])) * trainTime[trainsRetour[i][0]][stretch]
                trainsRetour[i][2] = stretch + 1
                waitingTime+=numbersBack+1


            #Trains to be popped for retour, because they are finished.
            if trainsRetour[i][2] - 1e-5 < 0:
                numberPopRetour += 1

        ##############################################################################3

        #This chunck uses insertion sort to sort the trainsTour and trainsRetour lists. They might not be sorted anymore, due to pushing back of trains.
        i = 0
        while i < len(trainsTour):
            j = i
            while j > 0 and trainsTour[j - 1][2] < trainsTour[j][2] - 1e-5:
                temp = trainsTour[j - 1]
                trainsTour[j - 1] = trainsTour[j]
                trainsTour[j] = temp
                j -= 1
            i += 1

        i = 0
        while i < len(trainsRetour):
            j = i
            while j > 0 and trainsRetour[j - 1][2] > trainsRetour[j][2] + 1e-5:
                temp = trainsRetour[j - 1]
                trainsRetour[j - 1] = trainsRetour[j]
                trainsRetour[j] = temp
                j -= 1
            i += 1

        #######################################################################################
        #Pops trains, adds their total run time to totalTime
        for i in range(numberPopTour):
            totalTime+=minutes-trainsTour[0][3]
            trainsTour.popleft()

        for i in range(numberPopRetour):
            totalTime += minutes - trainsRetour[0][3]
            trainsRetour.popleft()

        numberPopTour = 0
        numberPopRetour = 0

        #Resets the busy-labels
        for i in range(stations - 1):
            strechesTrain[i] = 0

    #Finally, at the end, runs through remaining trains and adding their time to the totalTime.
    for i in range(len(trainsTour)):
        totalTime+=time-trainsTour[i][3]


    for i in range(len(trainsRetour)):
        totalTime+=time-trainsRetour[i][3]

    return waitingTime

#This function does the same as the fitness-function, but also keeps track of their movement in a list to be plotted.
#This function is also rather big and messy, but is made with computation speed in mind. Have therefore reduced function calls.
def runTrains2(stations, time, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, passByTour, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains, trainTypesRetour):
    trainsTourPlot = np.zeros((numTrainTour, time))
    trainsRetourPlot = np.ones((numTrainRetour, time)) * sum(trainTime[0])

    strechesTrain = [0] * (stations - 1)
    trainsTour = deque()
    trainsRetour = deque()

    numberTour = 0
    numberRetour = 0

    numberPopTour = 0
    numberPopRetour = 0

    for minutes in range(time):
        #print(minutes)

        if startTimeOneTour == 0:
            numberTour += 1
            startTimeOneTour += frequencyOne
            trainsTour.append([0, numberTour, 0, minutes])

        if startTimeOneRetour == 0:
            numberRetour += 1
            startTimeOneRetour += frequencyOne
            trainsRetour.append([0, numberRetour, stations - 1, minutes])

        if startTimeTwoTour == 0:
            numberTour += 1
            startTimeTwoTour += frequencyTwo
            trainsTour.append([1, numberTour, 0, minutes])

        if startTimeTwoRetour == 0:
            numberRetour += 1
            startTimeTwoRetour += frequencyTwo
            trainsRetour.append([1, numberRetour, stations - 1, minutes])


        startTimeOneTour -= 1
        startTimeOneRetour -= 1
        startTimeTwoTour -= 1
        startTimeTwoRetour -= 1

#########################################################################

        for i in range(len(trainsTour)):
            stretch = int(trainsTour[i][2] + 1e-5)
            stretchMinus=int(trainsTour[i][2] - 1e-5)
            rate = trainTime[0][stretch]        #Rate is a multiplication factor to make the movements proportional to the distances between the stations.
            trainsTour[i][2] += 1 / trainTime[trainsTour[i][0]][stretch]
            trainsTourPlot[trainsTour[i][1] - 1][minutes] = trainsTourPlot[trainsTour[i][1] - 1][minutes - 1] + 1 / trainTime[trainsTour[i][0]][stretch] * rate
            newStretch = int(trainsTour[i][2] + 1e-5)
            for l in range(i, 0, -1):
                if trainsTour[i][2]  > trainsTour[l - 1][2]+1e-5:
                    if stretch == stretchMinus:
                        if 1 == passByTour[trainsTour[i][1] - 1][trainsTour[l - 1][1] - 1]:

                            trainsTour[l-1][2]=stretch

                            #This while-loop-mechanism goes back to alter the plot-list when a train is being pushed back to wait.
                            k=minutes
                            while trainsTourPlot[trainsTour[l-1][1] - 1][k] > sum(trainTime[0][0:stretch]) + 1e-5:
                                trainsTourPlot[trainsTour[l-1][1] - 1][k] = sum(trainTime[0][0:stretch])
                                k -= 1
                            numbersBack=(trainsTour[i][2]%1)*trainTime[trainsTour[i][0]][stretch]

                            for p in range(minutes-int(numbersBack+1e-5)+1,minutes+1):
                                trainsTour[l-1][2]+= 1 / trainTime[trainsTour[l-1][0]][stretch]
                                trainsTourPlot[trainsTour[l-1][1] - 1][p] = trainsTourPlot[trainsTour[l-1][1] - 1][p - 1] + 1 / trainTime[trainsTour[l-1][0]][stretch] * rate
                        else:
                            trainsTour[i][2] -= 1 / trainTime[trainsTour[i][0]][stretch]
                            newStretch=stretch
                            trainsTourPlot[trainsTour[i][1] - 1][minutes]=trainsTourPlot[trainsTour[i][1] - 1][minutes-1]
                            break
            if int(trainsTour[i][2]+1e-5) == stations - 1:
                numberPopTour += 1


            strechesTrain[stretch]=1



##############################################################################
        for i in range(len(trainsRetour)):
            stretch = int(trainsRetour[i][2] - 1e-5)
            cantMove = False
            if strechesTrain[stretch] == 1:
                for j in range(len(trainsTour) - 1, -1, -1):
                    if int(trainsTour[j][2] - 1e-5) == stretch:
                        if meetingTrains[trainsTour[j][1] - 1][trainsRetour[i][1] - 1] == 0:
                            cantMove=True

            if cantMove:
                continue
            else:
                if strechesTrain[stretch]==1:
                    strechesTrain[stretch] = 0
                    for j in range(len(trainsTour) - 1, -1, -1):
                        if int(trainsTour[j][2] - 1e-5) == stretch:
                            if int(trainsTour[j][2] + 1e-5) == stations - 1:
                                numberPopTour -= 1
                            trainsTour[j][2] = stretch


                            k = minutes
                            while trainsTourPlot[trainsTour[j][1] - 1][k] > sum(trainTime[0][0:stretch]) + 1e-5:
                                trainsTourPlot[trainsTour[j][1] - 1][k] = sum(trainTime[0][0:stretch])

                                k -= 1


###############################################################################
        for i in range(len(trainsRetour)):
            stretch=int(trainsRetour[i][2]-1e-5)
            stretchMinus=int(trainsRetour[i][2]+1e-5)
            rate = trainTime[0][stretch]

            if strechesTrain[stretch]==0:
                trainsRetour[i][2] -= 1 / trainTime[trainsRetour[i][0]][stretch]
                trainsRetourPlot[trainsRetour[i][1] - 1][minutes] = trainsRetourPlot[trainsRetour[i][1] - 1][minutes - 1] - 1 / trainTime[trainsRetour[i][0]][stretch] * rate
                newStretch = int(trainsRetour[i][2] - 1e-5)



                for l in range(i, 0, -1):
                    if trainsRetour[i][2]  < trainsRetour[l - 1][2]-1e-5:
                        if stretch == stretchMinus:
                            if 1 == passByTour[trainsRetour[l-1][1] - 1][trainsRetour[i][1] - 1]:

                                trainsRetour[l-1][2]=stretch+1
                                k=minutes
                                while trainsRetourPlot[trainsRetour[l-1][1] - 1][k] < sum(trainTime[0][0:stretch+1]) - 1e-5:
                                    trainsRetourPlot[trainsRetour[l-1][1] - 1][k] = sum(trainTime[0][0:stretch+1])
                                    k -= 1
                                numbersBack = (1-(trainsRetour[i][2] % 1)) * trainTime[trainsRetour[i][0]][stretch]
                                for p in range(minutes-int(numbersBack+1e-5)+1,minutes+1):
                                    trainsRetour[l-1][2]-= 1 / trainTime[trainsRetour[l-1][0]][stretch]
                                    trainsRetourPlot[trainsRetour[l-1][1] - 1][p] = trainsRetourPlot[trainsRetour[l-1][1] - 1][p - 1] - 1 / trainTime[trainsRetour[l-1][0]][stretch] * rate
                            else:
                                trainsRetour[i][2] += 1 / trainTime[trainsRetour[i][0]][stretch]
                                newStretch=stretch
                                trainsRetourPlot[trainsRetour[i][1] - 1][minutes]=trainsRetourPlot[trainsRetour[i][1] - 1][minutes-1]
                                break
            else:
                trainsRetour[i][2] = stretch + 1


                k = minutes-1
                while trainsRetourPlot[trainsRetour[i][1] - 1][k] < sum(trainTime[0][0:stretch + 1]) - 1e-5:
                    trainsRetourPlot[trainsRetour[i][1] - 1][k] = sum(trainTime[0][0:stretch + 1])
                    k -= 1
                trainsRetourPlot[trainsRetour[i][1] - 1][minutes] = sum(trainTime[0][0:stretch + 1])

            if trainsRetour[i][2] - 1e-5 < 0:
                numberPopRetour +=1


##############################################################################3

        i = 0
        while i < len(trainsTour):
            j = i
            while j > 0 and trainsTour[j - 1][2] < trainsTour[j][2] - 1e-5:
                temp = trainsTour[j - 1]
                trainsTour[j - 1] = trainsTour[j]
                trainsTour[j] = temp
                j -= 1
            i += 1

        i = 0
        while i < len(trainsRetour):
            j = i
            while j > 0 and trainsRetour[j - 1][2] > trainsRetour[j][2] + 1e-5:
                temp = trainsRetour[j - 1]
                trainsRetour[j - 1] = trainsRetour[j]
                trainsRetour[j] = temp
                j -= 1
            i += 1

#######################################################################################
        for i in range(numberPopTour):
            trainsTour.popleft()

        for i in range(numberPopRetour):
            trainsRetour.popleft()

        numberPopTour=0
        numberPopRetour=0

        for i in range(stations-1):
            strechesTrain[i]=0


    plotTrains(trainsTourPlot, time, numTrainTour, stations, 'tour', trainTime, trainTypesTour)
    plotTrains(trainsRetourPlot, time, numTrainRetour, stations, 'retour', trainTime, trainTypesRetour)
    plt.xlabel("Minutes")
    plt.ylabel("Distance")
    plt.yticks([])
    plt.show()