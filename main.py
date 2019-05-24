import numpy as np
import matplotlib.pyplot as plt
import trains as tr
import genetic as ge
import enfri as en
import time

#Figure explaining greedy choice
def plot_greedy_choice():
    station0=np.zeros(18)
    station1=np.ones(18)*3
    station2=np.ones(18)*6
    station3=np.ones(18)*9
    trainRetour=[9,9,9,9,9,8,7,6,5,4,3,2,1,0,0,0,0,0]
    train1=[0,1,2,3,3,3,3,3,3,3,3,4,5,6,7,8,9,9]
    train2=[0,1,2,3,4,5,6,6,6,6,6,6,6,6,7,8,9,9]
    train3=[0,1,2,3,4,5,6,6,7,8,9,9,9,9,9,9,9,9]
    x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    plt.figure()
    plt.plot(x,station0, c='black',linewidth=0.5)
    plt.plot(x, station1, c='black',linewidth=0.5)
    plt.plot(x, station2, c='black',linewidth=0.5)
    plt.plot(x, station3, c='black',linewidth=0.5)
    plt.plot(x,trainRetour, c='blue')
    plt.plot(x,train1, c='green')
    plt.plot(x, train2, c='green', linestyle='dashed')
    plt.plot(x, train3, c='green', linestyle='dashed')
    plt.xlabel("Minutes")
    plt.ylabel("Distance")
    plt.yticks([])
    plt.savefig("greedy_choice.pdf")
    plt.show()


#Computes many combinations of stuff. For initial impression of performance. Requires a lot of time
def plot_probability_to_all(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    probabilities = [0.001, 0.005, 0.01, 0.05]
    maxRank = [1.05, 1.1, 1.3, 1.7]
    replaceNumber = [2, 5, 8, 10]
    type="proportional"
    for i in replaceNumber:
        plt.figure()
        for j in probabilities:
            meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
            x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour,startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains,passByTour, j, numTrainTour, numTrainRetour, maxRank[0], i, type)
            plt.plot(x, y, label="probability=" + str(j))
        plt.legend()
        plt.title("Proportional, replaceNumber=" + str(i))
        plt.xlabel("Computation time")
        plt.ylabel("Fitness")
        plt.savefig("Proportional.replaceNumber" + str(i) + ".pdf")
        # plt.show()

    type = "rank"
    
    for i in maxRank:
        for j in replaceNumber:
            plt.figure()
            for k in probabilities:
                meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
                x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour, k, numTrainTour, numTrainRetour, i, j, type)
                plt.plot(x, y, label="probability="+str(k))

            plt.legend()
            plt.title("Rank, max="+str(i)+", replaceNumber="+str(j))
            plt.xlabel("Computation time")
            plt.ylabel("Fitness")
            plt.savefig("Rank.max"+str(i)+"replaceNumber"+str(j)+".pdf")
            #plt.show()

#Plots 5 different of each, to look at variance, and performance
def plot_prob2(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    probabilities = 0.01
    maxRank = 1.1
    replaceNumber = 2
    type = "rank"
    plt.figure()
    meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
    x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
    plt.plot(x, y, label="M=1.1, P=0.01, R=2", c='r')
    for i in range(4):
        meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
        x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour,startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime,meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
        plt.plot(x, y, c='r')

    maxRank = 1.7
    type = "rank"
    meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
    x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
    plt.plot(x, y, label="M=1.7, P=0.01, R=2", c='b')
    for i in range(4):
        meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
        x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour,
                              startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime,
                              meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
        plt.plot(x, y, c='b')

    probabilities = 0.05
    maxRank = 1.3
    replaceNumber = 5
    type = "rank"
    meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
    x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
    plt.plot(x, y, label="M=1.3, P=0.05, R=5", c='g')
    for i in range(4):
        meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
        x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
        plt.plot(x, y, c='g')



    plt.legend()
    #plt.title("Rank, probability=0.1, replaceNumber=" + str(i))
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("Different1")
    # plt.show()


#Probability analysis
def plot_Prob_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    probabilities = [0.001, 0.004, 0.007, 0.01, 0.03, 0.05]
    maxRank = 1.1
    replaceNumber = 2
    type = "rank"

    plt.figure()
    for i in probabilities:
        xi=0
        for j in range(5):
            meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
            x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour,startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime,meetingTrains, passByTour, i, numTrainTour, numTrainRetour, maxRank, replaceNumber, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="P=" + str(i))
    plt.legend()
    plt.title("Rank, M=1.1, R=2, n=10")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("Probability.pdf")
    # plt.show()

#Max analysis
def plot_max_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    probabilities = 0.01
    maxRank = [1.05, 1.1, 1.2, 1.4, 1.6, 1.8]
    replaceNumber = 2
    type = "rank"

    plt.figure()
    for i in maxRank:
        xi=0
        for j in range(5):
            meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
            x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour,startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime,meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, i, replaceNumber, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="M=" + str(i))
    plt.legend()
    plt.title("Rank, P=0.01, R=2, n=10")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("Max.pdf")
    # plt.show()

#Replacement analysis
def plot_replace_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    probabilities = 0.01
    maxRank = 1.1
    replaceNumber = [2,3,5,7,8,9]
    type = "rank"

    plt.figure()
    for i in replaceNumber:
        xi=0
        for j in range(5):
            meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
            x, y = ge.convergence(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour,startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime,meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, i, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="R=" + str(i))
    plt.legend()
    plt.title("Rank, P=0.01, M=1.1, n=10")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("Replace.pdf")
    # plt.show()

#number of individuals analysis
def plot_individuals_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    probabilities = 0.01
    maxRank = 1.1
    replaceNumber = 2
    individuals=[5,10,20,50,100]
    type = "rank"

    plt.figure()
    for i in individuals:
        xi=0
        for j in range(5):
            meetingTrains, passByTour = ge.initialPopulation(stations, i, numTrainTour, numTrainRetour)
            x, y = ge.convergence(runTime, i, stations, timeTotal, startTimeOneTour, startTimeTwoTour,startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime,meetingTrains, passByTour, probabilities, numTrainTour, numTrainRetour, maxRank, int(i*0.2), type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="n=" + str(i))
    plt.legend()
    plt.title("Rank, P=0.01, M=1.1, R=0.2")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("Individuals.pdf")
    # plt.show()

#T analysis with alpha1
def plot_T_alpha1_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    T = [100, 500, 1000, 5000,10000, 50000]
    k = 0.9
    type = 1
    plt.figure()
    for i in T:
        xi=0
        for j in range(5):
            x, y = en.run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, i, numTrainTour, numTrainRetour, k, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="T=" + str(i))
    plt.legend()
    plt.title("Annealing, k=0.9, Type=1")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("T1.pdf")
    # plt.show()

#T analysis with alpha2
def plot_T_alpha2_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    T = [100, 500, 1000, 5000,10000, 50000]
    k = 0.1
    type = 2
    plt.figure()
    for i in T:
        xi=0
        for j in range(5):
            x, y = en.run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, i, numTrainTour, numTrainRetour, k, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="T=" + str(i))
    plt.legend()
    plt.title("Annealing, k=0.1, Type=2")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("T2.pdf")
    # plt.show()

#k analysis with alpha1
def plot_k_alpha1_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    T = 50000
    k = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    type = 1
    plt.figure()
    for i in k:
        xi=0
        for j in range(5):
            x, y = en.run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, T, numTrainTour, numTrainRetour, i, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="k=" + str(i))
    plt.legend()
    plt.title("Annealing, T=5000, Type=1")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("k1.pdf")
    # plt.show()

#k analysis with alpha2
def plot_k_alpha2_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour):
    T = 50000
    k = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
    type = 2
    plt.figure()
    for i in k:
        xi=0
        for j in range(5):
            x, y = en.run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, T, numTrainTour, numTrainRetour, i, type)
            xi+=y
        xi/=5
        plt.plot(x, y, label="k=" + str(i))
    plt.legend()
    plt.title("Annealing, T=5000, Type=2")
    plt.xlabel("Seconds")
    plt.ylabel("Fitness")
    plt.savefig("k2.pdf")
    # plt.show()

#Gives an overview over simulated annealing. Requires a lot of time, gives overview of performance
def annealing_general(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour,
                            startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour,
                            numTrainRetour):
    T = [100, 1000, 10000]
    k=[0.7,0.8, 0.9 ,0.99]
    type=1
    for j in T:
        plt.figure()
        for l in k:
            x, y=en.run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, j, numTrainTour, numTrainRetour, l, type)
            plt.plot(x, y, label="k=" + str(l))
        plt.legend()
        plt.title("Annealing, T=" + str(j)+", type=1")
        plt.xlabel("Seconds")
        plt.ylabel("Fitness")
        plt.savefig("Annealing.T=" + str(j)+", type=1.pdf")
        # plt.show()

    T = [100, 1000, 10000]
    k = [0.001, 0.01, 0.1]
    type = 2
    for j in T:
        plt.figure()
        for l in k:
            x, y = en.run(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,
                          startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, j, numTrainTour, numTrainRetour, l,
                          type)
            plt.plot(x, y, label="k=" + str(l))
        plt.legend()
        plt.title("Annealing, T=" + str(j) + ", type=2")
        plt.xlabel("Seconds")
        plt.ylabel("Fitness")
        plt.savefig("Annealing.T=" + str(j) + ", type=2.pdf")
        # plt.show()

#Constructed problem with likely best solution, to check performance
def runToPerfectOne(runTime, individuals, probabilities, maxRank, replace,T,k, timeTotal):
    stations = 5
    startTimeOneTour = 0
    startTimeOneRetour = 1
    startTimeTwoTour = 400
    startTimeTwoRetour = 17
    frequencyOne = 26
    frequencyTwo = 26
    trainTime = [[3, 8, 8, 3], [4, 7, 15, 4]]

    numTrainTour, trainTypesTour = tr.calcNumTrain(startTimeOneTour, startTimeTwoTour, frequencyOne, frequencyTwo, timeTotal)
    numTrainRetour, trainTypesRetour = tr.calcNumTrain(startTimeOneRetour, startTimeTwoRetour, frequencyOne,
                                                       frequencyTwo, timeTotal)

    #Perform genetic convergence plot
    meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
    x1, y1, meetingTrains1, passByTour1= ge.convergence2(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,
                      startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour,
                      probabilities, numTrainTour, numTrainRetour, maxRank, replace, "rank")


    #Perform simulated annealing convergence plot
    x2, y2, meetingTrains2, passByTour2 = en.run2(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,
                  startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, T, numTrainTour, numTrainRetour, k,
                  1)

    #Assumed best solution
    passByTour = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    meetingTrains=[[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    #Fitness of best solution
    bestFitness=tr.fitness(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,
                     frequencyOne, frequencyTwo, trainTime, passByTour, meetingTrains)

    #Plotting
    plt.figure()
    plt.plot(x1, y1, label="Genetic algorithm")
    plt.plot(x2, y2, label="Simulated annealing")
    bestList=np.ones(len(x1))*bestFitness
    plt.plot(x1, bestList, label="Probably best solution")
    plt.xlabel("Computation time, seconds")
    plt.ylabel("Fitness")
    plt.title("Convergence plots for constructed problem, train run time "+str(timeTotal))
    plt.legend()
    plt.show()

    #More plotting
    tr.runTrains2(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,
                  frequencyOne,
                  frequencyTwo, trainTime, passByTour1, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains1,
                  trainTypesRetour)
    tr.runTrains2(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,
                  frequencyOne,
                  frequencyTwo, trainTime, passByTour2, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains2,
                  trainTypesRetour)
    tr.runTrains2(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,
                  frequencyOne,
                  frequencyTwo, trainTime, passByTour, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains,
                  trainTypesRetour)

#Makes convergence plots and solution plots for the standard problem
def runNormal(stations, startTimeOneTour, startTimeOneRetour, startTimeTwoTour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour, trainTypesTour, trainTypesRetour,runTime, individuals, probabilities, maxRank, replace,T,k, timeTotal):
    meetingTrains, passByTour = ge.initialPopulation(stations, individuals, numTrainTour, numTrainRetour)
    x1, y1, meetingTrains1, passByTour1= ge.convergence2(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,
                      startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, meetingTrains, passByTour,
                      probabilities, numTrainTour, numTrainRetour, maxRank, replace, "rank")



    x2, y2, meetingTrains2, passByTour2 = en.run2(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,
                  startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, T, numTrainTour, numTrainRetour, k,
                  1)

    plt.figure()
    plt.plot(x1, y1, label="Genetic algorithm")
    plt.plot(x2, y2, label="Simulated annealing")
    plt.xlabel("Computation time, seconds")
    plt.ylabel("Fitness")
    plt.title("Convergence plots for normal problem, train run time "+str(timeTotal))
    plt.legend()
    plt.show()

    tr.runTrains2(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,
                  frequencyOne,
                  frequencyTwo, trainTime, passByTour1, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains1,
                  trainTypesRetour)
    tr.runTrains2(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour,
                  frequencyOne,
                  frequencyTwo, trainTime, passByTour2, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains2,
                  trainTypesRetour)

def main():

    ######           CONSTANTS            #########
    individuals=10
    stations = 6
    timeTotal = 480
    startTimeOneTour = 0
    startTimeOneRetour = 5
    startTimeTwoTour = 10
    startTimeTwoRetour = 13
    frequencyOne = 25
    frequencyTwo = 30
    trainTime = [[3, 6, 7, 2, 4], [5, 12, 15, 3, 7]]

    numTrainTour, trainTypesTour = tr.calcNumTrain(startTimeOneTour, startTimeTwoTour, frequencyOne, frequencyTwo, timeTotal)
    numTrainRetour, trainTypesRetour = tr.calcNumTrain(startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo,timeTotal)

    ################################################

    #Runs and plots a random solution:
    #passByTour = np.random.randint(2, size=(numTrainTour, numTrainTour))
    #meetingTrains = np.random.randint(2, size=(numTrainTour, numTrainRetour))
    #tr.runTrains2(stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne,frequencyTwo, trainTime, passByTour, numTrainTour, numTrainRetour, trainTypesTour, meetingTrains,trainTypesRetour)

    #plots the visualization of argument that greedy choice works
    #plot_greedy_choice()

    #Set computational time for convergence plots
    runTime = 10

    #Plots many different combinations of stuff for genetic algoritm. Needs many hours depending on runtime
    #plot_probability_to_all(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    #Plots several times a small selection of combinations of parameters for genetic algorithm
    #plot_prob2(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour,numTrainRetour)

    #Plots for different probabolities, averaged over 5
    #plot_Prob_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour,numTrainRetour)

    #Plots for different Max, averaged over 5
    #plot_max_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour,startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    #Plots for different R, averaged over 5
    #plot_replace_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    #Plots for different n, averaged over 5
    #plot_individuals_analysis(runTime, individuals, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    #Plots many different combination of stuff, for annealing, requires alot of time depending on runtime
    #annealing_general(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    #Plots for different T, alpha1, averaged over 5
    #plot_T_alpha1_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    # Plots for different T, alpha2, averaged over 5
    #plot_T_alpha2_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    # Plots for different k, alpha1, averaged over 5
    #plot_k_alpha1_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    # Plots for different k, alpha2, averaged over 5
    #plot_k_alpha2_analysis(runTime, stations, timeTotal, startTimeOneTour, startTimeTwoTour, startTimeOneRetour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour)

    #Final constants
    probabilities=0.01
    maxRank=1.1
    replace=2
    individuals=10
    T=10000
    k=0.9
    timeTotal=200

    #This runs convergence plots and solutions to the constructed problem
    #runToPerfectOne(runTime, individuals, probabilities, maxRank, replace,T,k, timeTotal)

    #Runs convergence plots and solutions to standard problem
    #runNormal(stations, startTimeOneTour, startTimeOneRetour, startTimeTwoTour, startTimeTwoRetour, frequencyOne, frequencyTwo, trainTime, numTrainTour, numTrainRetour, trainTypesTour, trainTypesRetour, runTime,individuals, probabilities, maxRank, replace, T, k, timeTotal)


main()