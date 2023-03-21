import gameGUI

boardRowAmount = 6
boardColumnAmount = 7
connectAmountToWin = 4
defaultPlayers = ('Human', 'Computer')
# defaultPlayers = ('Human', 'Human')



def main():
    '''
    This is the function that actually runs the entire game
    '''
    gameGUI.menuGUI(boardRowAmount, boardColumnAmount, connectAmountToWin, defaultPlayers)


if __name__ == '__main__':
    main()
