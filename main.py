import gameGUI
import neuralNetTrainingPython as nnTrainPy6

boardRowAmount = 6
boardColumnAmount = 7
connectAmountToWin = 4
defaultPlayers = ('Human', 'Computer')
# defaultPlayers = ('Human', 'Human')


def main():
    gameGUI.menuGUI(boardRowAmount, boardColumnAmount, connectAmountToWin, defaultPlayers)


main()