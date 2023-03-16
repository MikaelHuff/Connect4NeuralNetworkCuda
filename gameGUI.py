import PySimpleGUI as sg
import time

import airand
import board
import neuralNetPlayingPython as nnPlay

guiTheme = 'LightBlue'
pieceColors = ['Black', 'White', 'Red']


def menuGUI(rowAmount, colAmount, amountToWin, players):
    layout = [[sg.Text('Row Amount'), sg.Push(), sg.In(size=(15, 10), key='_rowAmt_', default_text=str(rowAmount))],
              [sg.Text('Column Amount'), sg.Push(), sg.In(size=(15, 10), key='_colAmt_', default_text=str(colAmount))],
              [sg.Text('Amount to Win'), sg.Push(), sg.In(size=(15, 10), key='_winAmt_', default_text=str(amountToWin))],
              [sg.Text('Player 1'), sg.Push(), sg.Combo(['Human', 'Computer'], default_value=players[0], key='_P1_')],
              [sg.Text('Player 2'), sg.Push(), sg.Combo(['Human', 'Computer'], default_value=players[1], key='_P2_')],
              [sg.Push(), sg.Button('Start')],
              [sg.Push(), sg.Button('Close')]]
    window = sg.Window('Connect 4 Menu', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Close':
            break

        if event == 'Start':
            window.close()
            boardGUI(int(values['_rowAmt_']), int(values['_colAmt_']), int(values['_winAmt_']), (values['_P1_'], values['_P2_']))
            break

    window.close()


def boardGUI(rowAmount, colAmount, amountToWin, players):
    actionRow = [sg.Button(str(i + 1), size=(7, 2)) for i in range(colAmount)]
    boardRow = [[sg.Image('Images\\' + pieceColors[1] + '.png', size=(64, 64), key=('_board ' + str(i) + ',' + str(j) + '_')) for j in range(colAmount)] for i
                in range(rowAmount)]
    infoText = [sg.Text('', key='_text_')]
    menu = [sg.Button('Return to Menu'), sg.Button('Restart Game'), sg.Push(), sg.Button('Close')]

    gameState = board.Board(rowAmount, colAmount, amountToWin)
    gameOver = False

    if players[0] == 'Computer':
        infoText = [sg.Text('Please press restart game for the computer to start', key='_text_')]

    finalLayout = [actionRow, boardRow, infoText, menu]
    window = sg.Window('Connect 4', finalLayout)
    result = 0
    while True:
        event, values = window.read()
        window['_text_'].update('')

        try:
            if int(event) > 0 and not gameOver:
                if gameState.is_move_legal(int(event)-1):
                    height, result = gameState.update_board(int(event) - 1)
                    if height >= 0:
                        window['_board ' + str(height) + ',' + str(int(event) - 1) + '_'].update('Images\\' + pieceColors[-1*gameState._player + 1] + '.png', size=(64, 64), )
                else:
                    window['_text_'].update('Invalid Move: That column is full')
                    continue
                if result == -1:
                    gameOver = True
                    window['_text_'].update(str(pieceColors[-1 * gameState._player + 1]) + ' wins')
                if result == -2:
                    gameOver = True
                    window['_text_'].update('No more moves, game is a tie')
                window.refresh()
        except Exception as e:
            if type(e) == AssertionError:
                raise
            pass



        if event == sg.WIN_CLOSED or event == 'Close':
            break

        if event == 'Restart Game':
            gameState = board.Board(rowAmount, colAmount, amountToWin)
            for i in range(rowAmount):
                for j in range(colAmount):
                    window['_board ' + str(i) + ',' + str(j) + '_'].update('Images\\' + pieceColors[1] + '.png', size=(64, 64))
            gameOver = False
            window['_text_'].update('')

        if event == 'Return to Menu':
            window.close()
            menuGUI(rowAmount, colAmount, amountToWin, players)
            break

        while players[int((-1*gameState.get_player+ 1)/2)] == 'Computer' and not gameOver:
            if players[int((gameState.get_player+ 1)/2)] == 'Human':
                time.sleep(.1)
            move = nnPlay.find_best_move(gameState)
            height, result = gameState.update_board(move)
            window['_board ' + str(height) + ',' + str(int(move)) + '_'].update('Images\\' + pieceColors[-1*gameState._player + 1] + '.png', size=(64, 64))
            if result == -1:
                gameOver = True
                window['_text_'].update(str(pieceColors[-1 * gameState.get_player + 1]) + ' wins')
            if result == -2:
                gameOver = True
                window['_text_'].update('No more moves, game is a tie')
            window.refresh()

    window.close()
