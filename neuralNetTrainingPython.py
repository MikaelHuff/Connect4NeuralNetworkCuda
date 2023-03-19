import numpy as np
import os
import time

import board
import neuralNetPlayingPython as nnPlayPy

# import run_pybind as pybi

boardRowAmount = 6
boardColumnAmount = 7
connectAmountToWin = 4
hiddenNodeAmount = 100
alpha = .01
epsilon = 1e-50
dataDir = nnPlayPy.get_data_dir()


def create_Network():
    '''
    Creates a network initialized randomely
    :return: A network
    '''
    inputWeights = np.random.uniform(low=-1, high=1, size=(boardRowAmount * boardColumnAmount, hiddenNodeAmount))
    hiddenWeights = np.random.uniform(low=-1, high=1, size=(hiddenNodeAmount, boardColumnAmount))
    return [inputWeights, hiddenWeights]


def save_network(network, gamesPlayed):
    '''
    Saves the given network with a file name corresponding to amount of games played
    :param network: Network to be saved
    :param gamesPlayed: The amount of games played divided by 10000
    '''
    np.savez(dataDir + 'Network Weights' + str(gamesPlayed) + '.npz', network[0], network[1])


def train_network(networkLocation=0, playMethod='python', updateMethod='vectorized'):
    '''
    Trains a neural network
    :param networkLocation: Location of the file, This is given by the amount of games played, so 0 means new network
    :param playMethod: Whether to evaluate the game state and network using python or cuda
    :param updateMethod: the method for which to update the network
    '''
    if networkLocation > 0:
        network = nnPlayPy.read_network(networkLocation)
    else:
        network = create_Network()

    gameNumber = networkLocation * 10000 + 1
    timeVec = np.zeros(3)
    # While True, training occurs
    while True:
        # This block of code is to play the game. It first creates a new game.
        gameOngoing = True
        gameRecord = []
        gameState = board.Board(boardRowAmount, boardColumnAmount, connectAmountToWin)
        turnAmount = 0
        # While the game has not ended, it continously plays and stores the data from each move
        while gameOngoing:
            turnData = nnPlayPy.get_turn_info(gameState, network)
            if turnData[0] == -1:
                # An illegal move was made
                assert (turnData[0] >= 0)
                pass
            result = gameState.update_board(turnData[0])[1]
            gameRecord.append(turnData)
            turnAmount += 1
            if result < 0:
                gameOngoing = False

        # Once the game has finished, it is time to label the data for our network to use. First we find the winner.
        if result == -1:
            winner = -1 * gameState.get_player
        elif result == -2:
            winner = 0
        # Here we calculate the matrix representing the goal. Each row is a turn, and each column value is the expected value for that move given the board state of that turn.
        moveLabels = np.zeros((turnAmount, boardColumnAmount))
        for turn in range(1, turnAmount):
            for j in range(boardColumnAmount):
                if j == gameRecord[turn][0]:
                    moveLabels[turn, j] = get_scaling_factor(turn, len(moveLabels)) * (winner * (turn % 2 - .5) * 2)

        # Updating the weights. Choose any of the following methods. Test does all 3, ensures matching outcomes, and checks time comparisons.
        if updateMethod == 'individual':
            network = update_weights_individual(network, gameRecord, moveLabels)
        elif updateMethod == 'vectorized':
            network = update_weights_vectorized(network, gameRecord, moveLabels)
        elif updateMethod == 'cuda':
            network = update_weights_cuda(network, gameRecord, moveLabels)
        elif updateMethod == 'test':
            start = time.time()
            network1 = update_weights_individual(network, gameRecord, moveLabels)
            time1 = time.time() - start
            start = time.time()
            network2 = update_weights_vectorized(network, gameRecord, moveLabels)
            time2 = time.time() - start
            start = time.time()
            network3 = update_weights_cuda(network, gameRecord, moveLabels)
            time3 = time.time() - start

            allTimes = [time1, time2, time3]
            if timeVec.max() == 0:
                timeVec = np.array(allTimes)
            else:
                for i in range(3):
                    timeVec[i] = (timeVec[i] * (gameNumber - 1) + allTimes[i]) / gameNumber
            print('Average run times for individual: ', timeVec[0], ', for vectorized: ', timeVec[1], ', and for cuda: ', timeVec[2])
            assert (network1 == network2 == network3), "Not matching values"
            network = network2

        # Increase the gameNumber and deal with information. Every 1000 games, outputs values to see information on current network. Every 10,000 games, the network is saved.
        gameNumber += 1
        if gameNumber % 1000 == 0:
            print(gameNumber, np.sum(network[0]), network[0].max() - network[0].min(), np.sum(network[1]), network[1].max() - network[1].min())
        if gameNumber % 10000 == 0:
            saveValue = int(gameNumber / 10000)
            save_network(network, saveValue)
            if os.path.exists(dataDir + "Network Weights" + str(saveValue - 2) + ".npz"):
                os.remove(dataDir + "Network Weights" + str(saveValue - 2) + ".npz")
            print("Network saved")


def update_weights_individual(network, gameRecord, labels):
    '''
    Updates the weights for the given network. This function does it by using for loops to individually update each single weight
    :param network: The neural network
    :param gameRecord: The record of game values as the game was played
    :param labels: The values the network should have achieved
    :return: An updated network
    '''
    inputWeightsDelta = np.zeros(np.shape(network[0]))
    hiddenWeightsDelta = np.zeros(np.shape(network[1]))
    turnAmount = len(gameRecord)
    # The first set of loops, is used to update the weights for each hidden layer node
    for turn in range(1, turnAmount):
        for outputNode in range(boardColumnAmount):
            outputError = np.sum(gameRecord[turn][2][:, outputNode]) - labels[turn, outputNode]
            for node in range(hiddenNodeAmount):
                impact = np.abs(gameRecord[turn][2][node, outputNode]) / np.sum(np.abs(gameRecord[turn][2][:, outputNode]))
                hiddenWeightsDelta[node, outputNode] += impact * outputError
    network[1] -= alpha * hiddenWeightsDelta / turnAmount

    # The second set of loops, is used to update the weights for each input layer node
    for turn in range(1, turnAmount):
        for outputNode in range(hiddenNodeAmount):
            outputError = np.sum(hiddenWeightsDelta, axis=1)[outputNode]
            for node in range(boardColumnAmount * boardRowAmount):
                impact = gameRecord[turn][1][node, outputNode] / (np.sum(np.abs(gameRecord[turn][1][:, outputNode])))
                inputWeightsDelta[node, outputNode] += impact * outputError
    network[0] -= alpha * inputWeightsDelta / turnAmount

    network[0] = network[0] / np.abs(network[0]).max()
    network[1] = network[1] / np.abs(network[1]).max()
    return network


def update_weights_vectorized(network, gameRecord, labels):
    '''
    Updates the weights for the given network. This function does it by using for loops to individually update each single weight
    :param network: The neural network
    :param gameRecord: The record of game values as the game was played
    :param labels: The values the network should have achieved
    :return: An updated network
    '''
    inputWeightsDelta = np.zeros(np.shape(network[0]))
    hiddenWeightsDelta = np.zeros(np.shape(network[1]))
    turnAmount = len(gameRecord)
    # The first set of loops, is used to update the weights for each hidden layer node
    for turn in range(1, turnAmount):
        outputErrorMat = np.sum(gameRecord[turn][2]) - labels[turn, :]
        impactMat = np.divide(np.abs(gameRecord[turn][2]), (np.sum(np.abs(gameRecord[turn][2]), axis=0) + epsilon))
        hiddenWeightsDelta += np.multiply(impactMat, outputErrorMat) / turnAmount
    network[1] -= alpha * hiddenWeightsDelta  # (1-alpha) *

    # The second set of loops, is used to update the weights for each input layer node
    for turn in range(1, turnAmount):
        outputErrorMat = np.sum(hiddenWeightsDelta, axis=1)
        impactMat = np.divide(np.abs(gameRecord[turn][1]), np.sum(np.abs(gameRecord[turn][1]), axis=0) + epsilon)
        inputWeightsDelta += np.multiply(impactMat, outputErrorMat) / turnAmount
    network[0] -= alpha * inputWeightsDelta  # (1-alpha) *

    network[0] = network[0] / np.abs(network[0]).max()
    network[1] = network[1] / np.abs(network[1]).max()
    return network


def update_weights_cuda(network, gameRecord, labels):
    '''
    Updates the weights for the given network. This function does it by using for loops to individually update each single weight
    :param network: The neural network
    :param gameRecord: The record of game values as the game was played
    :param labels: The values the network should have achieved
    :return: An updated network
    '''
    # network = pybi.update_weights(network, gameRecord, labels)
    return network


def get_scaling_factor(idx, amount):
    return (idx + 1) / amount


# train_network()

if __name__ == '__main__':
    # train_network_vectorized(1289)
    train_network(updateMethod='test')
