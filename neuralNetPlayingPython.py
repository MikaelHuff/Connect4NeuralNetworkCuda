import numpy as np
import os

#import run_pybind as pybi

dataDir = __file__[0:-25]+'data\\'
def get_data_dir():
    return dataDir

def read_network(saveValue=-1):
    if saveValue == -1:
        saveValue = os.listdir(dataDir)[-1][15:-4]
    npzfile = np.load(dataDir + 'Network Weights' + str(saveValue) + '.npz')
    return [npzfile[npzfile.files[0]], npzfile[npzfile.files[1]]]


def get_turn_info(board, network=None, playMethod = 'python'):
    if network is None:
        network = read_network()

    if playMethod == 'cuda':
        #return pybi.get_turn_info(board, network)
        pass

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
        if val == max and np.random.uniform() > tieOdds and board.is_move_legal(move):
            tieOdds *= tieAmt / (tieAmt + 1)
            tieAmt += 1
            max = val
            bestMove = move

    if not board.is_move_legal(bestMove):
        return [-1*bestMove, np.reshape(boardVector,np.shape(boardVector)[0]), np.expand_dims(inputOutput,axis=1), np.expand_dims(finalOutput,axis=1)]
    else:
        return [bestMove, np.reshape(boardVector,np.shape(boardVector)[0]), np.expand_dims(inputOutput,axis=1), np.expand_dims(finalOutput,axis=1)]


def find_best_move(board, method = 'python'):
     return get_turn_info(board, playMethod=method)[0]


# print(read_network())
