import numpy as np


class Board:
    '''
    A board class for the game to be played on
    '''

    def __init__(self, rows, columns, amountToWin):
        '''
        Initiates class
        :param rows: amount of rows for the board to have
        :param columns: amount of columns for the board to have
        '''
        assert (isinstance(rows, int)), "Amount of rows must be an integer"
        assert (isinstance(columns, int)), "Amount of columns must be an integer"
        assert (isinstance(amountToWin, int)), "Amount to win must be an integer"
        self._rows = rows
        self._cols = columns
        self._winAm = amountToWin
        self._board = np.zeros([rows, columns])
        self._player = 1
        self._turns = 0

    def update_board(self, move):
        '''
        Updates the board with a given move
        :param move: Move made on board
        '''
        assert (isinstance(move, int)), "Move must be an integer"
        assert (0 <= move < self._cols), "Move out of bounds"
        assert (self.is_move_legal(move)), "Not a legal move"

        # make move
        moveHeight = 0
        for row in range(self._rows):
            if self._board[self._rows - row - 1, move] == 0:
                self._board[self._rows - row - 1, move] = self._player
                moveHeight = self._rows - row - 1
                break

        self._player *= -1
        self._turns +=1
        # check if game has been won
        if self.check_win():
            return moveHeight, -1
        # check if board is full
        if self.check_board_full():
            return moveHeight, -2

        return moveHeight, 0

    def is_move_legal(self, move):
        if 0 in self._board[:, move]:
            return True
        return False

    def check_win(self):
        '''
        Checks if the previous move led to a winning board. Goes through the directons starting with up and cycling
        through the possibilities clockwise. Indexing starts at (0,0) in bottom left corner

        :return: True if win, False if not
        '''

        # Check vertical win
        for j in range(self._cols):
            for i in range(self._rows - (self._winAm - 1)):
                for n in range(self._winAm):
                    if self._board[i + n, j] != -1 * self._player:
                        break
                else:
                    return True

        # Check horizontal win
        for i in range(self._rows):
            for j in range(self._cols - (self._winAm - 1)):
                for n in range(self._winAm):
                    if self._board[i, j + n] != -1 * self._player:
                        break
                else:
                    return True

        # Check diagonal win :  top left to bottom right
        for i in range(self._rows - (self._winAm - 1)):
            for j in range(self._cols - (self._winAm - 1)):
                for n in range(self._winAm):
                    if self._board[i + n, j + n] != -1 * self._player:
                        break
                else:
                    return True

        # Check diagonal win: bottom left to top right
        for i in range(self._rows - (self._winAm - 1)):
            for j in range(self._cols - (self._winAm - 1)):
                for n in range(self._winAm):
                    if self._board[i + (self._winAm - 1) - n, j + n] != -1 * self._player:
                        break
                else:
                    return True

        # No win detected
        return False

    def check_board_full(self):
        '''
        Sees if any moves are left to be played
        :return: False if at least one empty slot. True if all full
        '''
        if 0 in self._board:
            return False
        return True

    def get_board_vector(self):
        return np.reshape(self._board, (self._rows*self._cols,1))

    @property
    def get_player(self):
        return self._player
