import numpy as np
import os

import board
import neuralNetPlayingPython as nnPlayPy
import airand

boardRowAmount = 6
boardColumnAmount = 7
connectAmountToWin = 4
hiddenNodeAmount = 100
alpha = .01
epsilon = 1e-50


def create_Network():
    inputWeights = np.random.uniform(low=-1, high=1, size=(boardRowAmount * boardColumnAmount, hiddenNodeAmount))
    hiddenWeights = np.random.uniform(low=-1, high=1, size=(hiddenNodeAmount, boardColumnAmount))
    return [inputWeights, hiddenWeights]


def save_network(network, gamesPlayed):
    np.savez('data\\Network Weights' + str(gamesPlayed) + '.npz', network[0], network[1])


def train_network(networkLocation=0, playMethod='python', updateMethod='vectorized'):
    if networkLocation > 0:
        network = nnPlayPy.read_network(networkLocation)
    else:
        network = create_Network()

    gameNumber = networkLocation*10000 + 1
    while True:
        # play a game
        gameOngoing = True
        gameRecord = []
        gameState = board.Board(boardRowAmount, boardColumnAmount, connectAmountToWin)
        turnAmount = 0
        while gameOngoing:
            turnData = nnPlayPy.get_turn_info(gameState, network)
            if turnData[0] == -1:
                # An illegal move was made
                assert(turnData[0] >= 0)
                pass
            result = gameState.update_board(turnData[0])[1]
            gameRecord.append(turnData)
            turnAmount += 1
            # otherMove = airand.get_move2(gameState)
            # result = gameState.update_board(otherMove)[1]
            if result < 0:
                gameOngoing = False

        # judge and label the game
        if result == -1:
            winner = -1 * gameState.get_player
        elif result == -2:
            winner = 0
        moveLabels = np.zeros((turnAmount, boardColumnAmount))
        for turn in range(1,turnAmount):
            for j in range(boardColumnAmount):
                if j == gameRecord[turn][0]:
                    moveLabels[turn, j] = get_scaling_factor(turn, len(moveLabels)) * (winner * ( turn % 2- .5) * 2)

        # update the weights

        if updateMethod == 'individual':
            network = update_weights_individual(network, gameRecord, moveLabels)
        elif updateMethod == 'vectorized':
            network = update_weights_vectorized(network, gameRecord, moveLabels)
        elif updateMethod == 'cuda':
            pass
        elif updateMethod == 'test':
            network1 = update_weights_individual(network, gameRecord, moveLabels)
            network2 = update_weights_vectorized(network, gameRecord, moveLabels)
            # network3 = update_weights_cuda(network, gameRecord, moveLabels)
            assert(network1 == network2), "Not matching values"
            network = network2


        # print
        gameNumber += 1
        print(gameNumber)
        if gameNumber % 1000 == 0:
            # print(gameNumber)
            print(gameNumber, np.sum(network[0]), network[0].max()-network[0].min(), np.sum(network[1]), network[1].max()-network[1].min())
        if gameNumber % 10000 == 0:
            saveValue = int(gameNumber/10000)
            save_network(network, saveValue)
            if os.path.exists("data\\Network Weights"+str(saveValue-2)+".npz"):
                os.remove("data\\Network Weights"+str(saveValue-2)+".npz")
            print("Network saved")





def update_weights_individual(network, gameRecord, labels):
    inputWeightsDelta = np.zeros(np.shape(network[0]))
    hiddenWeightsDelta = np.zeros(np.shape(network[1]))
    turnAmount = len(gameRecord)
    for turn in range(1, turnAmount):
        for outputNode in range(boardColumnAmount):
            outputError = np.sum(gameRecord[turn][2][:, outputNode]) - labels[turn, outputNode]
            for node in range(hiddenNodeAmount):
                impact = np.abs(gameRecord[turn][2][node, outputNode]) / np.sum(np.abs(gameRecord[turn][2][:, outputNode]))
                hiddenWeightsDelta[node, outputNode] += impact * outputError
    network[1] -= alpha * hiddenWeightsDelta / turnAmount

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
    inputWeightsDelta = np.zeros(np.shape(network[0]))
    hiddenWeightsDelta = np.zeros(np.shape(network[1]))
    turnAmount = len(gameRecord)
    for turn in range(1, turnAmount):
        outputErrorMat = np.sum(gameRecord[turn][2]) - labels[turn, :]
        impactMat = np.divide(np.abs(gameRecord[turn][2]), (np.sum(np.abs(gameRecord[turn][2]), axis=0) + epsilon))
        hiddenWeightsDelta += np.multiply(impactMat, outputErrorMat) / turnAmount
    network[1] -= alpha * hiddenWeightsDelta  # (1-alpha) *

    for turn in range(1, turnAmount):
        outputErrorMat = np.sum(hiddenWeightsDelta, axis=1)
        impactMat = np.divide(np.abs(gameRecord[turn][1]), np.sum(np.abs(gameRecord[turn][1]), axis=0) + epsilon)
        inputWeightsDelta += np.multiply(impactMat, outputErrorMat) / turnAmount
    network[0] -= alpha * inputWeightsDelta  # (1-alpha) *

    network[0] = network[0] / np.abs(network[0]).max()
    network[1] = network[1] / np.abs(network[1]).max()
    return network


def get_scaling_factor(idx, amount):
    return (idx + 1) / amount


# train_network()

if __name__ == '__main__':
    # train_network_vectorized(1289)
    train_network(updateMethod='test')
