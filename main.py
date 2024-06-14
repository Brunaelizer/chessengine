import chess
import chess.engine
import random
import numpy
import ia
import chess.svg
import chessboard
from chessboard import display
from time import sleep

# Abre o modelo e seta como variável global
model = ia.open_model()

# Lista para tradução das casas
squares = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}


# Gerar um tabuleiro aleatório
def get_board():
    # Gera o tabuleiro
    board = chess.Board()
    # Escolhe a profundidade
    depth = 100

    # Cria um tabuleiro com posições aleatórias, seguindo regra da profundidade
    for i in range(depth):
        moves = list(board.legal_moves)
        random_move = random.choice(moves)
        board.push(random_move)

        if board.is_game_over():
            print("Game over")
            break

    return board


# Traduzir o código da casa para numero de posição
def get_square_index(square):
    letter = chess.square_name(square)
    number = 8 - int(letter[1]), squares[letter[0]]
    return number


# Separar as dimensões do tabuleiro
def split_dimension(board):
    matrix_board = numpy.zeros((14, 8, 8), dtype=numpy.int8)

    # Adiciona os tipos das peças nas matrizes
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            # Gera tupla de coordenadas
            i = numpy.unravel_index(square, (8, 8))
            matrix_board[piece - 1][7 - i[0]][i[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            i = numpy.unravel_index(square, (8, 8))
            matrix_board[piece + 5][7 - i[0]][i[1]] = 1

    # Adicionar os ataques e os movimentos legais
    turn_new = board.turn

    # Brancas
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = get_square_index(move.to_square)
        matrix_board[12][i][j] = 1

    # Pretas
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = get_square_index(move.to_square)
        matrix_board[13][i][j] = 1
    board.turn = turn_new

    return matrix_board


# Método para a árvore de minmax
def minimax(board, depth, alpha, beta, is_max):
    if depth == 0 or board.is_game_over():
        matrix_board = split_dimension(board)
        matrix_board = numpy.expand_dims(matrix_board, 0)
        return model.predict(matrix_board)[0][0]

    # Verifica maximização do jogador
    if is_max:
        max_eva = -numpy.inf
        for move in board.legal_moves:
            board.push(move)
            eva = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eva = max(max_eva, eva)
            alpha = max(alpha, eva)
            if beta <= alpha:
                break
        return max_eva
    else:
        min_eva = numpy.inf
        for move in board.legal_moves:
            board.push(move)
            eva = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eva = min(min_eva, eva)
            beta = min(beta, eva)
            if beta <= alpha:
                break
        return min_eva


# Movimentação a partir da IA
def get_move(board, depth):
    max_eva = -numpy.inf

    for move in board.legal_moves:
        board.push(move)
        eva = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
        board.pop()
        if eva > max_eva:
            max_eva = eva
            max_move = move

    return max_move


# Método para dar inicio no jogo em si
def start_game(board):
    # Visualizar o tabuleiro
    view_tabuleiro = display.start()
    display.update(board.fen(), view_tabuleiro)

    # Chama a IA do Stockfish
    with chess.engine.SimpleEngine.popen_uci('stockfish/stockfish_15_x64_avx2.exe') as sf:
        while True:
            # Jogada da IA
            print("\nIA\n")
            move = get_move(board, 1)
            board.push(move)
            print(f'\n{board}')
            if board.is_game_over():
                get_outcome(board)
                sleep(100)
                break

            display.update(board.fen(), view_tabuleiro)

            print("\nStockfish\n")
            # Jogada do Stockfish
            move = sf.analyse(board, chess.engine.Limit(time=1), info=chess.engine.INFO_PV)['pv'][0]
            board.push(move)
            print(f'\n{board}')
            if board.is_game_over():
                get_outcome(board)
                sleep(100)
                break

            display.update(board.fen(), view_tabuleiro)


def get_outcome(board):
    print("\nGame over")
    print(board.outcome())

    if board.is_stalemate():
        print("Empate por afogamento")
    elif board.is_check():
        print("Checkmate")
    elif board.is_insufficient_material():
        print("Empate por material")



def main():
    # Gerar tabuleiro aleatorio
    board = get_board()
    print("Tabuleiro gerado:")
    print(board)


    # Treinar IA
    # ia.train()

    # Abrir modelo
    start_game(board)


if __name__ == '__main__':
    main()
