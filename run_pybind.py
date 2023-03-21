import sys

sys.path.append("C:/Users/mkhuff/Downloads/Python2Cuda_example/build/src/pybind11_cpp_examples/Release")
sys.path.append("C:/Users/mkhuff/Downloads/Python2Cuda_example/build/src/pybind11_cuda_examples/Release")

import cu_matrix_add as cudaM

import numpy as np

epsilon = 1e-5
alpha = .01


def get_turn_info(board, network=None):
    if network is None:
        network = read_network()

    boardVector = board.get_board_vector()
    inputValues = cudaM.mmul(network[0], cudaM.mtile(boardVector, 70)).transpose()
    inputOutput = cudaM.msum2(inputValues)
    finalValues = cudaM.mmul(network[1], cudaM.mtile(inputOutput, 7))
    finalOutput = cudaM.msum(finalValues)

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
        return [-1 * bestMove, np.reshape(boardVector, np.shape(boardVector)[0]), inputOutput, finalOutput]
    else:
        return [bestMove, np.reshape(boardVector, np.shape(boardVector)[0]), inputOutput, finalOutput]


def find_best_move(board):
    return get_turn_info(board)[0]


def update_weights(gameRecord, labels):
    dims = [len(gameRecord[0][1]), len(gameRecord[0][2]), len(gameRecord[0][3])]
    inputWeightsDelta = np.zeros((dims[0], dims[1]))
    hiddenWeightsDelta = np.zeros((dims[1], dims[2]))
    turnAmount = len(gameRecord)
    for turn in range(turnAmount):
        outputErrorMat = cudaM.msub(gameRecord[turn][3], np.expand_dims(labels[turn], axis=1))
        impactMat = gameRecord[turn][2]
        hiddenWeightsDelta += cudaM.mcoef(cudaM.mmatmul(impactMat, cudaM.mtrans(outputErrorMat)), alpha / turnAmount)

    for turn in range(turnAmount):
        # print("1",np.shape(hiddenWeightsDelta))
        outputErrorMat = cudaM.msum2(hiddenWeightsDelta)
        # print("2",np.shape(outputErrorMat))
        # print("3",np.shape(gameRecord[turn][1]))
        impactMat = np.expand_dims(gameRecord[turn][1], axis=1)
        # print("4",np.shape(impactMat))
        # print("5",np.shape(cudaM.mtrans(outputErrorMat)))
        inputWeightsDelta += cudaM.mcoef(cudaM.mmatmul(impactMat, cudaM.mtrans(outputErrorMat)), alpha / turnAmount)
        # print("6",np.shape(inputWeightsDelta))
        # input()

    return [inputWeightsDelta, hiddenWeightsDelta]


if __name__ == '__main__':
    A = np.random.uniform(low=-10, high=10, size=(5, 5))
    B = np.random.uniform(low=-10, high=10, size=(5, 5))
    print(A)
    C = cudaM.mcoef(A, 3)
    print(C)
    input()
    assert (np.sum(np.abs(A + B - cudaM.madd(A, B))) < epsilon), f"cuda addition not accurate, value 1: {A + B} \n value 2:{cudaM.madd(A, B)}"
    assert (np.sum(np.abs(A - B - cudaM.msub(A, B))) < epsilon), f"cuda subtraction not accurate, value 1: {A - B} \n value 2:{cudaM.msub(A, B)}"
    coef = B[0, 0]
    assert (np.sum(np.abs(coef * A - cudaM.mcoef(A, coef))) < epsilon), f"cuda coef scaling not accurate, coef: {coef} \n, value 1: {coef * A} \n value 2:{cudaM.mcoef(A, coef)}"
    C = np.random.uniform(low=-10, high=10, size=(5, 1))
    intCoef = int(np.random.uniform(low=0, high=10, size=(1, 1)))
    assert (np.sum(np.abs(
        np.tile(C, intCoef).transpose() - cudaM.mtile(C, intCoef))) < epsilon), f"cuda tiling not accurate, rep: {intCoef}\n, value 1: {np.tile(C, intCoef)} \n value 2:{cudaM.mtile(C, intCoef)}"
    assert (np.sum(np.abs(np.abs(A) - cudaM.mabs(A))) < epsilon), f"cuda absolute value not accurate, value 1: {np.abs(A)} \n value 2:{cudaM.mabs(A)}"
    assert (np.sum(np.abs(np.sum(A, axis=0) - cudaM.msum(A))) < epsilon), f"cuda row sum not accurate, value 1: {np.sum(A, axis=0)} \n value 2:{cudaM.msum(A)}"
    assert (np.sum(np.abs(np.sum(A, axis=1) - cudaM.msum2(A))) < epsilon), f"cuda col sum not accurate, value 1: {np.sum(A, axis=1)} \n value 2:{cudaM.msum2(A)}"
    assert (np.sum(np.abs(np.multiply(A, B) - cudaM.mmul(A, B))) < epsilon), f"cuda multiplication not accurate, value 1: {np.multiply(A, B)} \n value 2:{cudaM.mmul(A, B)}"
    assert (np.sum(np.abs(np.divide(A, B) - cudaM.mdiv(A, B))) < epsilon), f"cuda divide not accurate, value 1: {np.divide(A, B)} \n value 2:{cudaM.mdiv(A, B)}"
    assert (np.sum(np.abs(A.transpose() - cudaM.mtrans(A))) < epsilon), f"cuda transpose not accurate, value 1: {A.transpose()} \n value 2:{cudaM.mtrans(A)}"
    assert (np.sum(np.abs((A + coef) - cudaM.maddCons(A, coef))) < epsilon), f"cuda matrix plus coefficient not accurate, value 1: {(A + coef)} \n value 2:{cudaM.maddCons(A, coef)}"
    D = np.random.uniform(low=-10, high=10, size=(1, 7))
    assert (np.sum(np.abs(np.matmul(C, D) - cudaM.mmatmul(C, D))) < epsilon), f"cuda addition not accurate, value 1: {np.matmul(C, D)} \n value 2:{cudaM.mmatmul(C, D)}"
    print("All tests complete")
