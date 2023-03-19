import sys
sys.path.append("C:/Users/mkhuff/Downloads/Python2Cuda_example/build/src/pybind11_cpp_examples/Release")
sys.path.append("C:/Users/mkhuff/Downloads/Python2Cuda_example/build/src/pybind11_cuda_examples/Release")

import cu_matrix_add as cudaM

import numpy as npgi

epsilon = 1 # 1e-50
alpha = .01

def update_weights(network, gameRecord, labels):
    inputWeightsDelta = np.zeros(np.shape(network[0]))
    hiddenWeightsDelta = np.zeros(np.shape(network[1]))
    turnAmount = len(gameRecord)
    for turn in range(1, turnAmount):
        outputErrorMat1 = cudaM.mtrans(cudaM.msum(gameRecord[turn][2]))
        outputErrorVect = cudaM.madd(outputErrorMat1, np.reshape(labels[turn], (7,1)),1,-1)
        outputErrorMat = cudaM.mtile(outputErrorVect,100)
        impactMatNum = cudaM.mabs(gameRecord[turn][2])
        impactMatDen = cudaM.mtile(cudaM.mtrans(cudaM.msum(cudaM.maddCons(impactMatNum, epsilon))),100)
        impactMat = cudaM.mdiv(impactMatNum, impactMatDen)
        turnDelta = cudaM.mmul(impactMat, outputErrorMat)
        hiddenWeightsDelta = cudaM.madd(hiddenWeightsDelta, turnDelta, 1, 1/turnAmount )
    network[1] = cudaM.madd(network[1], hiddenWeightsDelta, 1, -alpha)

    for turn in range(1, turnAmount):
        outputErrorVect = cudaM.mtrans(cudaM.msum(cudaM.mtrans(hiddenWeightsDelta)))
        outputErrorMat = cudaM.mtile(outputErrorVect,42)
        impactMatNum = cudaM.mabs(gameRecord[turn][1])
        impactMatDen = cudaM.mtile(cudaM.mtrans(cudaM.msum(cudaM.maddCons(impactMatNum, epsilon))),42)
        impactMat = cudaM.mdiv(impactMatNum, impactMatDen)
        turnDelta = cudaM.mmul(impactMat,outputErrorMat)
        inputWeightsDelta = cudaM.madd(inputWeightsDelta, turnDelta, 1, 1/turnAmount )
    network[0] = cudaM.madd(network[0], inputWeightsDelta, 1, -alpha)

    network[0] = network[0] / np.abs(network[0]).max()
    network[1] = network[1] / np.abs(network[1]).max()
    return network

