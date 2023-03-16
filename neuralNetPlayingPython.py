import numpy as np
import os

def read_network(saveValue=-1):
    if saveValue == -1:
        saveValue = os.listdir('data')[-1][15:-4]
    npzfile = np.load('data\\Network Weights' + str(saveValue) + '.npz')
    return [npzfile[npzfile.files[0]], npzfile[npzfile.files[1]]]


def get_turn_info(board, network=None):
    if network is None:
        network = read_network()

    boardVector = board.get_board_vector()
    inputValues = np.multiply(network[0], boardVector).transpose()
    inputOutput = np.sum(inputValues, axis=1)
    finalValues = np.multiply(network[1].transpose(), inputOutput)
    finalOutput = np.sum(finalValues, axis=1)

    max = -1 * np.inf
    bestMove = -1
    tieOdds = .5
    tieAmt = 2
    for move, val in enumerate(finalOutput):
        if val > max and board.is_move_legal(move):
            max = val
            bestMove = move
        if val == max and np.random.uniform() > tieOdds:
            tieOdds *= tieAmt / (tieAmt + 1)
            tieAmt += 1
            max = val
            bestMove = move

    if not board.is_move_legal(bestMove):
        return [-1, inputValues, finalValues]
    else:
        return [bestMove, inputValues.transpose(), finalValues.transpose()]


def find_best_move(board):
    return get_turn_info(board)[0]

# print(read_network())
