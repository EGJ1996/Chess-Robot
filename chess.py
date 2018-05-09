from __future__ import print_function
import random
import time
import math
class State:
        def __init__(self, state_dict = None):
                if (state_dict != None):
                        self.board = state_dict['board']
                        self.turn = state_dict['turn']
                        self.current_move = state_dict['current_move']
                        self.white_king_loc = state_dict['white_king_loc']
                        self.black_king_loc = state_dict['black_king_loc']
                        self.white_castle = state_dict['white_castle']
                        self.black_castle = state_dict['black_castle']
                        self.en_passant = state_dict['en_passant']
                        self.en_passant_loc = state_dict['en_passant_loc']
                        
                        self.opponent_attack_dict = state_dict['opponent_attack_dict']
                        self.legal_move_list = state_dict['legal_move_list']
                        
                        self.white_value = state_dict['white_value']
                        self.black_value = state_dict['black_value']
                        self.value = state_dict['value']
                else:
                        self.blank_state()
        
        def blank_state(self):                  
                size = 8
                pieces = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
                
                self.board = [['· ' for y in range(size)] for x in range(size)]          
                for i in range(size):
                        self.board[0][i] = pieces[i]
                        self.board[1][i] = 'p'
                        self.board[6][i] = 'P'
                        self.board[7][i] = pieces[i].upper()
                
                self.turn = 'white'
                self.current_move = None
                self.white_king_loc = (7, 4)
                self.black_king_loc = (0, 4)
                self.white_castle = [True, True]
                self.black_castle = [True, True]
                self.en_passant = []
                self.en_passant_loc = []
                
                self.opponent_attack_dict = {}
                self.legal_move_list = []
                
                self.white_value = 0
                self.black_value = 0
                self.value = 0
class Chess:
        def __init__(self):
                self.size = 8           
                self.states = []
                
                self.files = {'a':'a', 'b':'b', 'c':'c', 'd':'d', 'e':'e', 'f':'f', 'g':'g', 'h':'h'}
                self.ranks = {8:0, 7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7}
                self.coordinate_to_notation_files = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
                self.coordinate_to_notation_ranks = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
                self.notation_to_coordinate_files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
                self.notation_to_coordinate_ranks = {8: 0, 7: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6, 1: 7}
                
                self.reverse_rank = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
                
                self.pawn_table = [[  0,  0,  0,  0,  0,  0,  0,  0],
                                                   [ 50, 50, 50, 50, 50, 50, 50, 50],
                                                   [ 10, 10, 20, 30, 30, 20, 10, 10],
                                                   [  5,  5, 10, 25, 25, 10,  5,  5],
                                                   [  0,  0,  0, 20, 20,  0,  0,  0],
                                                   [  5, -5,-10,  0,  0,-10, -5,  5],
                                                   [  5, 10, 10,-20,-20, 10, 10,  5],
                                                   [  0,  0,  0,  0,  0,  0,  0,  0],]
                
                self.knight_table = [[-50,-40,-30,-30,-30,-30,-40,-50],
                                                         [-40,-20,  0,  0,  0,  0,-20,-40],
                                                         [-30,  0, 10, 15, 15, 10,  0,-30],
                                                         [-30,  5, 15, 20, 20, 15,  5,-30],
                                                         [-30,  0, 15, 20, 20, 15,  0,-30],
                                                         [-30,  5, 10, 15, 15, 10,  5,-30],
                                                         [-40,-20,  0,  5,  5,  0,-20,-40],
                                                         [-50,-40,-30,-30,-30,-30,-40,-50],]
                                                         
                self.bishop_table = [[-20,-10,-10,-10,-10,-10,-10,-20],
                                                         [-10,  0,  0,  0,  0,  0,  0,-10],
                                                         [-10,  0,  5, 10, 10,  5,  0,-10],
                                                         [-10,  5,  5, 10, 10,  5,  5,-10],
                                                         [-10,  0, 10, 10, 10, 10,  0,-10],
                                                         [-10, 10, 10, 10, 10, 10, 10,-10],
                                                         [-10,  5,  0,  0,  0,  0,  5,-10],
                                                         [-20,-10,-10,-10,-10,-10,-10,-20],]
                                                        
                self.rook_table = [[  0,  0,  0,  0,  0,  0,  0,  0],
                                                   [  5, 10, 10, 10, 10, 10, 10,  5],
                                                   [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                   [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                   [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                   [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                   [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                   [  0,  0,  0,  5,  5,  0,  0,  0],]
                                                   
                self.queen_table = [[-20,-10,-10, -5, -5,-10,-10,-20],
                                                        [-10,  0,  0,  0,  0,  0,  0,-10],
                                                        [-10,  0,  5,  5,  5,  5,  0,-10],
                                                        [ -5,  0,  5,  5,  5,  5,  0, -5],
                                                        [  0,  0,  5,  5,  5,  5,  0, -5],
                                                        [-10,  0,  5,  5,  5,  5,  0,-10],
                                                        [-10,  0,  0,  0,  0,  0,  0,-10],
                                                        [-20,-10,-10, -5, -5, -5,-10,-20],]
                                                        
                self.king_table = [[-30,-40,-40,-50,-50,-40,-40,-30],
                                                   [-30,-40,-40,-50,-50,-40,-40,-30],
                                                   [-30,-40,-40,-50,-50,-40,-40,-30],
                                                   [-30,-40,-40,-50,-50,-40,-40,-30],
                                                   [-20,-30,-30,-40,-40,-30,-30,-20],
                                                   [-10,-20,-20,-20,-20,-20,-20,-10],
                                                   [ 20, 20,  0,  0,  0,  0, 20, 20],
                                                   [ 20, 30, 10,  0,  0, 10, 30, 20],]
                

                self.pawn_opp_table = [[  0,  0,  0,  0,  0,  0,  0,  0],
                                                           [ 50, 50, 50, 50, 50, 50, 50, 50],
                                                           [ 10, 10, 20, 30, 30, 20, 10, 10],
                                                           [  5,  5, 10, 25, 25, 10,  5,  5],
                                                           [  0,  0,  0, 20, 20,  0,  0,  0],
                                                           [  5, -5,-10,  0,  0,-10, -5,  5],
                                                           [  5, 10, 10,-20,-20, 10, 10,  5],
                                                           [  0,  0,  0,  0,  0,  0,  0,  0],]          
                self.pawn_opp_table.reverse()
                
                self.knight_opp_table = [[-50,-40,-30,-30,-30,-30,-40,-50],
                                                                 [-40,-20,  0,  0,  0,  0,-20,-40],
                                                                 [-30,  0, 10, 15, 15, 10,  0,-30],
                                                                 [-30,  5, 15, 20, 20, 15,  5,-30],
                                                                 [-30,  0, 15, 20, 20, 15,  0,-30],
                                                                 [-30,  5, 10, 15, 15, 10,  5,-30],
                                                                 [-40,-20,  0,  5,  5,  0,-20,-40],
                                                                 [-50,-40,-30,-30,-30,-30,-40,-50],]
                self.knight_opp_table.reverse()
                                                         
                self.bishop_opp_table = [[-20,-10,-10,-10,-10,-10,-10,-20],
                                                                 [-10,  0,  0,  0,  0,  0,  0,-10],
                                                                 [-10,  0,  5, 10, 10,  5,  0,-10],
                                                                 [-10,  5,  5, 10, 10,  5,  5,-10],
                                                                 [-10,  0, 10, 10, 10, 10,  0,-10],
                                                                 [-10, 10, 10, 10, 10, 10, 10,-10],
                                                                 [-10,  5,  0,  0,  0,  0,  5,-10],
                                                                 [-20,-10,-10,-10,-10,-10,-10,-20],]
                self.bishop_opp_table.reverse()
                                                        
                self.rook_opp_table = [[  0,  0,  0,  0,  0,  0,  0,  0],
                                                           [  5, 10, 10, 10, 10, 10, 10,  5],
                                                           [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                           [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                           [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                           [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                           [ -5,  0,  0,  0,  0,  0,  0, -5],
                                                           [  0,  0,  0,  5,  5,  0,  0,  0],]
                self.rook_opp_table.reverse()
                                                   
                self.queen_opp_table = [[-20,-10,-10, -5, -5,-10,-10,-20],
                                                                [-10,  0,  0,  0,  0,  0,  0,-10],
                                                                [-10,  0,  5,  5,  5,  5,  0,-10],
                                                                [ -5,  0,  5,  5,  5,  5,  0, -5],
                                                                [  0,  0,  5,  5,  5,  5,  0, -5],
                                                                [-10,  0,  5,  5,  5,  5,  0,-10],
                                                                [-10,  0,  0,  0,  0,  0,  0,-10],
                                                                [-20,-10,-10, -5, -5, -5,-10,-20],]
                self.queen_opp_table.reverse()
                                                        
                self.king_opp_table = [[-30,-40,-40,-50,-50,-40,-40,-30],
                                                           [-30,-40,-40,-50,-50,-40,-40,-30],
                                                           [-30,-40,-40,-50,-50,-40,-40,-30],
                                                           [-30,-40,-40,-50,-50,-40,-40,-30],
                                                           [-20,-30,-30,-40,-40,-30,-30,-20],
                                                           [-10,-20,-20,-20,-20,-20,-20,-10],
                                                           [ 20, 20,  0,  0,  0,  0, 20, 20],
                                                           [ 20, 30, 10,  0,  0, 10, 30, 20],]
                self.king_opp_table.reverse()           
                
                self.piece_values = {'R': 500, 'N': 320, 'B': 330, 'Q': 900, 'K': 20000, 'P': 100, '· ': 0,
                                                         'r': 500, 'n': 320, 'b': 330, 'q': 900, 'k': 20000, 'p': 100}
                
                self.square_values = {'R': self.rook_table, 'N': self.knight_table, 'B': self.bishop_table, 
                                                         'Q': self.queen_table, 'K': self.king_table, 'P': self.pawn_table,
                                                         'r': self.rook_opp_table, 'n': self.knight_opp_table, 'b': self.bishop_opp_table, 
                                                         'q': self.queen_opp_table, 'k': self.king_opp_table, 'p': self.pawn_opp_table,}
                
                self.directions = {                     
                        'P': ((-1,0), (-2,0), (-1,-1), (-1,1)),
                        'R': ((-1,0), (0,1), (1,0), (0,-1)),
                        'N': ((-2,1), (-1,2), (1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1)),
                        'B': ((-1,1), (1,1), (1,-1), (-1,-1)),
                        'Q': ((-1,0), (0,1), (1,0), (0,-1), (-1,1), (1,1), (1,-1), (-1,-1)), 
                        'K': ((-1,0), (0,1), (1,0), (0,-1), (-1,1), (1,1), (1,-1), (-1,-1)),
                        'p': ((-1,0), (-2,0), (-1,-1), (-1,1)),
                        'r': ((-1,0), (0,1), (1,0), (0,-1)),
                        'n': ((-2,1), (-1,2), (1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1)),
                        'b': ((-1,1), (1,1), (1,-1), (-1,-1)),
                        'q': ((-1,0), (0,1), (1,0), (0,-1), (-1,1), (1,1), (1,-1), (-1,-1)), 
                        'k': ((-1,0), (0,1), (1,0), (0,-1), (-1,1), (1,1), (1,-1), (-1,-1))
                        }
                
                self.piece_color = {
                        'P': 'white', 'R': 'white', 'N': 'white', 'B': 'white', 'Q': 'white', 'K': 'white',
                        'p': 'black', 'r': 'black', 'n': 'black', 'b': 'black', 'q': 'black', 'k': 'black',
                        '· ': None
                        }
                        
                state = State()
                self.states.append(state)
                self.state = self.states[-1]
                self.value=0
                white_value = 0
                black_value = 0
                value = 0
                for j, row in enumerate(self.state.board):
                                for i, piece in enumerate(row):
                                        if (piece.isupper()):                                   
                                                white_value += self.piece_values[piece] + self.square_values[piece][j][i]
                                        elif (piece.islower()):
                                                black_value += self.piece_values[piece] + self.square_values[piece][j][i]
                                        
                value = white_value-black_value         
                
                self.state.white_value = white_value
                self.state.black_value = black_value
                self.state.value = value
                
        def gen_moves(self):
                self.fulltime = 0.0     
                if (self.state.turn == 'white'):
                        castle = self.state.white_castle
                        color = 'white'
                        opp_color = 'black'
                else:
                        castle = self.state.black_castle
                        color = 'black'
                        opp_color = 'white'
                for j, row in enumerate(self.state.board):
                        for i, piece in enumerate(row):
                                if self.piece_color[piece] != color:
                                        continue
                                for direction in self.directions[piece]:
                                        for k in range(1,8):
                                                y = j + k*direction[0]
                                                x = i + k*direction[1]                                  
                                                
                                                if ((x < 0) or (x > 7) or (y < 0) or (y > 7)):
                                                        break
                                                
                                                newTile = self.state.board[y][x]
                                                if (self.piece_color[newTile] == color):
                                                        break
                                                if (piece.upper() == 'P'):
                                                        if ((direction in ((-1,0), (-2,0))) and (newTile != '· ')):
                                                                break
                                                        if ((direction == (-2,0)) and ((j != 6) or (self.state.board[j-1][i] != '· '))):
                                                                break
                                                        if ((direction in ((-1,-1), (-1,1))) and (newTile == '· ') and (((j,i) not in self.state.en_passant) or ((y,x) not in self.state.en_passant_loc))):
                                                                break
                                                        
                                                move = ((j,i), (y,x))
                                                yield(move)
                                                if ((piece.upper() in 'PNK') or (self.piece_color[newTile] == opp_color)):
                                                        break                                               
                                                if ((j,i) == (7,0) and (y,x) == (7, 3) and (self.state.board[7][4] == 'K') and (castle[0] == True)):
                                                        move = ((7, 4), (7, 2))
                                                        yield(move)
                                                if (((j,i) == (7,7))and (y,x) == (7, 5) and (self.state.board[7][4] == 'K') and (castle[1] == True)):
                                                        move = ((7, 4), (7, 6))
                                                        yield(move)
        def gen_check_attacks(self):
                move_dict = {}
                
                if (self.state.turn == 'white'):
                        castle = self.state.white_castle
                        color = 'white'
                        opp_color = 'black'
                else:
                        castle = self.state.black_castle
                        color = 'black'
                        opp_color = 'white'
                for j, row in enumerate(self.state.board):
                        for i, piece in enumerate(row):
                                if self.piece_color[piece] != opp_color: 
                                        continue
                                for direction in self.directions[piece]:
                                        pin_count = 0
                                        pin_position = None
                                        en_passant_move = False
                                        for k in range(1,8):
                                                y = j + k*-direction[0]
                                                x = i + k*direction[1]                                  
                                                
                                                if ((x < 0) or (x > 7) or (y < 0) or (y > 7)):
                                                        break
                                                
                                                newTile = self.state.board[y][x]
                                                
                                                if ((piece.upper() == 'P') and (direction in ((-1,0), (-2,0)))):
                                                        break
                                                        
                                                move = ((j,i), (y,x))
                                                input_direction = (-direction[0], direction[1])                                         
                                                
                                                if (self.piece_color[newTile] == opp_color):
                                                        data = [move, input_direction, pin_count, pin_position]
                                                        en_passant_check = (move[1][0] - 1, move[1][1])
                                                        if ((pin_count == 0 or pin_count == 1) and (en_passant_check in self.state.en_passant_loc) and (direction == (0,1) or direction == (0, -1))):
                                                                en_passant_move = self.state.en_passant_loc
                                                                if (move[1] not in move_dict):
                                                                        move_dict[move[1]] = []
                                                                        move_dict[move[1]].append(data)
                                                                else: 
                                                                        move_dict[move[1]].append(data)
                                                                continue
                                                        else:
                                                                en_passant_move = False
                                                                if (move[1] not in move_dict):
                                                                        move_dict[move[1]] = []
                                                                        move_dict[move[1]].append(data)
                                                                else: 
                                                                        move_dict[move[1]].append(data) 
                                                                break
                                                if (piece.upper() in 'PNK'):
                                                        data = [move, input_direction, pin_count, pin_position]                                                         
                                                        
                                                        en_passant_move = False
                                                        if (move[1] not in move_dict):
                                                                move_dict[move[1]] = []
                                                                move_dict[move[1]].append(data)
                                                        else: 
                                                                move_dict[move[1]].append(data)                                                         
                                                        break
                                                        
                                                if (self.piece_color[newTile] == color and newTile.upper() != 'K'):
                                                        pin_count += 1                                                  
                                                        if (pin_count == 1):
                                                                pin_position = (y,x)
                                                        else:
                                                                pin_position = None

                                                data = [move, input_direction, pin_count, pin_position]
                                                if (en_passant_move != False):
                                                        data.append(en_passant_move)
                                                                                                
                                                if (move[1] not in move_dict):
                                                        move_dict[move[1]] = []
                                                        move_dict[move[1]].append(data)
                                                else: 
                                                        move_dict[move[1]].append(data)
                                                        
                return(move_dict)
                
        #Check to see if your king is in check
        #data[0] = the current move to attack the king
        #data[1] = the direction of the attacking or pinned piece piece
        #data[2] = the number of pinned pieces for the move
        #data[3] = the location of the pinned piece
        #data[4] = the en passant location if there is one
        def king_in_check(self, move):
                                
                king_position = None
                if (self.state.turn == 'white'):
                        king_position = self.state.white_king_loc                                       
                elif (self.state.turn == 'black'):
                        king_position = self.state.black_king_loc               
                attack_dict = self.state.opponent_attack_dict
                king_target = king_position in attack_dict                                     
                in_check = False
                double_check = False            
                if (king_target):                       
                        pinned_list = attack_dict[king_position]
                        check_count = 0
                        for data in pinned_list:
                                if (data[2] == 0):
                                        check_count += 1
                        if (check_count == 1):
                                in_check = True
                        elif (check_count >= 2):
                                in_check = True
                                double_check = True
                if (not king_target):           
                        if (move[0] == king_position):
                                if (move[1] in attack_dict):
                                        for data in attack_dict[move[1]]: 
                                                if (data[2] == 0):
                                                        return(True)
                                        return(False)                                           
                                else: return(False)                             
                        else: return(False)
                elif (king_target and double_check == False):                   
                        if (move[0] == king_position):
                                if (move[1] in attack_dict):
                                        for data in attack_dict[move[1]]:
                                                if (data[2] == 0):
                                                        return(True)
                                        return(False)                                           
                                else: 
                                        return(False)
                        move_difference = [move[1][0]-king_position[0], move[1][1]-king_position[1]]                    
                        
                        divisor = 0
                        if (abs(move_difference[0]) >= abs(move_difference[1])):
                                divisor = abs(move_difference[0])
                                if (abs(move_difference[0]) == 2 and abs(move_difference[1]) == 1):
                                        divisor = 1
                        elif (abs(move_difference[0]) < abs(move_difference[1])):
                                divisor = abs(move_difference[1])
                                if (abs(move_difference[0]) == 1 and abs(move_difference[1]) == 2):
                                        divisor = 1
                        
                        if (divisor == 0):
                                direction = (0, 0)
                        else:
                                y = move_difference[0]/divisor
                                x = move_difference[1]/divisor                  
                                direction = (-y, -x)
                        pinned_list = attack_dict[king_position]
                        if in_check == False:
                                for data in pinned_list:
                                        if (data[2] == 1 and data[3] == move[0]):
                                                
                                                if (direction == data[1]):
                                                        return(False)
                                                else:
                                                        if (len(data) == 5 and move[1] not in data[4]):
                                                                return(False)
                                                        elif(len(data) == 5 and move[1] in data[4]):
                                                                return(True)
                                                        else:
                                                                return(True)                    
                                return(False)
                        elif (in_check == True):                              
                                for data in pinned_list:
                                        if (data[2] == 1 and data[3] == move[0]):
                                                if (direction != data[1]):
                                                        return(True)
                                for data in pinned_list:
                                        if (data[1] == direction and data[2] == 0):
                                
                                                check_difference = [data[0][0][0] - data[0][1][0], data[0][0][1] - data[0][1][1]]
                                                if (abs(move_difference[0]) <= abs(check_difference[0]) and abs(move_difference[1]) <= abs(check_difference[1])):
                                                        return(False)
                        return(True)
                elif (in_check == True and double_check == True):
                        if (move[0] == king_position):
                                if (move[1] in attack_dict):
                                        for data in attack_dict[move[1]]: 
                                                if (data[2] == 0):
                                                        return(True)
                                        return(False)                                           
                                else: return(False)
                        else: return(True)      
                
                return(None)
                
        def king_ray_check(self):
                in_check = False
                if (self.state.turn == 'white'):
                        king_position = self.state.white_king_loc
                        color = 'white'
                        opp_color = 'black'
                else:
                        king_position = self.state.black_king_loc
                        color = 'black'
                        opp_color = 'white'                     
                                        
                piece = 'K'
                j = king_position[0]
                i = king_position[1]
                
                for direction in self.directions[piece]:
                        for k in range(1,8):
                                
                                y = j + k*direction[0]
                                x = i + k*direction[1]                                  
                                
                                if ((x < 0) or (x > 7) or (y < 0) or (y > 7)):
                                        break
                                
                                newTile = self.state.board[y][x]
                                if (self.piece_color[newTile] == color):
                                        break
                                                        
                                if (newTile.upper() == 'P' and direction in ((-1,1), (-1,-1)) and k == 1):
                                        in_check = True                                 
                        
                                if (newTile.upper() == 'B' and direction in ((-1,1), (1,1), (1,-1), (-1,-1))):
                                        in_check = True 
                                        
                                if (newTile.upper() == 'R' and direction in ((-1,0), (0,1), (1,0), (0,-1))):
                                        in_check = True 
                                        
                                if (newTile.upper() == 'Q' and direction in ((-1,0), (0,1), (1,0), (0,-1), (-1,1), (1,1), (1,-1), (-1,-1))):
                                        in_check = True 
                                        
                piece = 'N'
                for direction in self.directions[piece]:
                        
                        y = j + direction[0]
                        x = i + direction[1]                                    
                        
                        if ((x < 0) or (x > 7) or (y < 0) or (y > 7)):
                                continue
                        
                        newTile = self.state.board[y][x]
                        if (newTile.upper() == 'N' and self.piece_color[newTile] == opp_color):
                                in_check = True
                        
                return(in_check)

        def format_move(self, move):
        
                if (len(move) != 4):
                        print('Input move in the form such as: e2e4')
                        return(False)
                        
                if ((move[0] not in self.files) or (move[2] not in self.files)):
                        print('Input move in the form such as: e2e4')
                        return(False)
                        
                if ((int(move[1]) not in self.ranks) or (int(move[3]) not in self.ranks)):
                        print('Input move in the form such as: e2e4')
                        return(False)
                
                coordinateMove = self.notation_to_coordinate(move)
                return(coordinateMove)
        
        def move_white_piece(self, move):
                current_move = self.coordinate_to_notation(move)
                current = move[0]
                new = move[1]   
                
                new_board = [row[:] for row in self.state.board] 

                p = new_board[current[0]][current[1]]
                q = new_board[new[0]][new[1]]
                
                white_value = self.state.white_value + (self.square_values[p][new[0]][new[1]] - self.square_values[p][current[0]][current[1]])
                black_value = self.state.black_value
                
                if (q in self.square_values):
                        black_value -= (self.piece_values[q] + self.square_values[q][new[0]][new[1]])
                        
                player_castle = self.state.white_castle
                opponent_castle = self.state.black_castle               
                white_king_loc = self.state.white_king_loc
                black_king_loc = self.state.black_king_loc              
                en_passant = []
                en_passant_loc = []
                
                new_board[new[0]][new[1]] = p
                new_board[current[0]][current[1]] = '· '         
                
                if (current == (7, 0)):
                        player_castle = [False, player_castle[1]]
                elif (current == (7, 7)):
                        player_castle = [player_castle[0], False]
                        
                if (new == (0, 0)):
                        opponent_castle = [False, opponent_castle[1]]
                elif (new == (0, 7)):
                        opponent_castle = [opponent_castle[0], False]
                
                if (p == 'K'):
                        player_castle = [False, False]
                        if (abs(new[1] - current[1]) == 2):
                        
                                if (new == (7, 2)):
                                        new_board[7][3] = 'R'
                                        new_board[7][0] = '· '
                                        
                                        white_value += self.square_values['R'][7][3] - self.square_values['R'][7][0]
                
                                if (new == (7,6)):
                                        new_board[7][5] = 'R'
                                        new_board[7][7] = '· '
                                        
                                        white_value += self.square_values['R'][7][5] - self.square_values['R'][7][7]
                        
                        white_king_loc = (new[0], new[1])
                        
                if (p == 'P'):
                
                        if (new[0] == 0):
                                new_board[new[0]][new[1]] = 'Q'
                                
                                white_value += self.piece_values['Q'] + self.square_values['Q'][new[0]][new[1]] - self.piece_values['P']
                                
                        if (abs(new[0]-current[0]) == 2):
                                if (new[1]-1 >= 0):
                                        en_passant.append((new[0], new[1]-1))
                                        en_passant_loc.append((new[0]+1, new[1]))
                                if (new[1]+1 <= 7):
                                        en_passant.append((new[0], new[1]+1))
                                        en_passant_loc.append((new[0]+1, new[1]))
                                        
                        ep_check = (new[0]-current[0], new[1]-current[1])
                        if ((ep_check == (-1, -1) or ep_check == (-1, 1)) and (q == '· ')):
                                        if (ep_check == (-1, -1)):                                                      
                                                new_board[current[0]][current[1]-1] = '· '
                                                
                                                black_value -= self.piece_values['p'] + self.square_values['p'][current[0]][current[1]-1]
                                        if (ep_check == (-1, 1)):                                                       
                                                new_board[current[0]][current[1]+1] = '· '
                                                
                                                black_value -= self.piece_values['p'] + self.square_values['p'][current[0]][current[1]+1]                                                       
                
                value = white_value-black_value
                new_board.reverse()     
                
                state_dict = {}
                state_dict['turn'] = 'black'    
                state_dict['white_castle'] = player_castle
                state_dict['black_castle'] = opponent_castle                                    
                state_dict['legal_move_list'] = []
                state_dict['opponent_attack_dict'] = {} 
                
                for i, coordinate in enumerate(en_passant):
                        en_passant[i] = (7-coordinate[0], coordinate[1])                        
                for i, coordinate in enumerate(en_passant_loc): 
                        en_passant_loc[i] = (7-coordinate[0], coordinate[1])
                        
                white_king_loc = (7-white_king_loc[0], white_king_loc[1])
                black_king_loc = (7-black_king_loc[0], black_king_loc[1])               
                state_dict['white_king_loc'] = white_king_loc
                state_dict['black_king_loc'] = black_king_loc
                state_dict['en_passant'] = en_passant
                state_dict['en_passant_loc'] = en_passant_loc
                state_dict['current_move'] = current_move       
                state_dict['board'] = new_board         
                state_dict['white_value'] = white_value
                state_dict['black_value'] = black_value
                state_dict['value'] = value
                
                state = State(state_dict)
                self.states.append(state)
                self.state = self.states[-1]            
                return state
                
        def move_black_piece(self, move):
                current_move = self.coordinate_to_notation(move)
                current = move[0]
                new = move[1]   
                
                new_board = [row[:] for row in self.state.board] 
                
                p = new_board[current[0]][current[1]]
                q = new_board[new[0]][new[1]]
                
                black_value = self.state.black_value + (self.square_values[p.upper()][new[0]][new[1]] - self.square_values[p.upper()][current[0]][current[1]])
                white_value = self.state.white_value
                
                if (q in self.square_values):
                        white_value -= (self.piece_values[q] + self.square_values[q.lower()][new[0]][new[1]])
        
                player_castle = self.state.black_castle
                opponent_castle = self.state.white_castle               
                white_king_loc = self.state.white_king_loc
                black_king_loc = self.state.black_king_loc              
                en_passant = []
                en_passant_loc = []
                
                new_board[new[0]][new[1]] = p
                new_board[current[0]][current[1]] = '· '         
                
                if (current == (7, 0)):
                        player_castle = [False, player_castle[1]]
                elif (current == (7, 7)):
                        player_castle = [player_castle[0], False]
                        
                if (new == (0, 0)):
                        opponent_castle = [False, opponent_castle[1]]
                elif (new == (0, 7)):
                        opponent_castle = [opponent_castle[0], False]
                
                if (p == 'k'):                  
                        player_castle = [False, False]
                        if (abs(new[1] - current[1]) == 2):                             
                                
                                if (new == (7, 2)):
                                        new_board[7][3] = 'r'
                                        new_board[7][0] = '· '
                                        
                                        black_value += self.square_values['R'][7][3] - self.square_values['R'][7][0]
                                if (new == (7,6)):
                                        new_board[7][5] = 'r'
                                        new_board[7][7] = '· '
                                        
                                        black_value += self.square_values['R'][7][5] - self.square_values['R'][7][7]
                                        
                        black_king_loc = (new[0], new[1])
                
                if (p == 'p'):
                        if (new[0] == 0):
                                new_board[new[0]][new[1]] = 'q'
                                
                                black_value += self.piece_values['q'] + self.square_values['Q'][new[0]][new[1]] - self.piece_values['p']
                                
                        if (abs(new[0]-current[0]) == 2):
                                if (new[1]-1 >= 0):
                                        en_passant.append((new[0], new[1]-1))
                                        en_passant_loc.append((new[0]+1, new[1]))
                                if (new[1]+1 <= 7):
                                        en_passant.append((new[0], new[1]+1))
                                        en_passant_loc.append((new[0]+1, new[1]))
                        
                        ep_check = (new[0]-current[0], new[1]-current[1])
                        if ((ep_check == (-1, -1) or ep_check == (-1, 1)) and (q == '· ')):
                                        if (ep_check == (-1, -1)):
                                                new_board[current[0]][current[1]-1] = '· '
                                                
                                                white_value -= self.piece_values['P'] + self.square_values['p'][current[0]][current[1]-1]
                                        if (ep_check == (-1, 1)):
                                                new_board[current[0]][current[1]+1] = '· '
                                                
                                                white_value -= self.piece_values['P'] + self.square_values['p'][current[0]][current[1]+1]
                
                value = white_value-black_value
                new_board.reverse()
                
                state_dict = {}
                state_dict['turn'] = 'white'    
                state_dict['white_castle'] = opponent_castle
                state_dict['black_castle'] = player_castle                                      
                state_dict['legal_move_list'] = []
                state_dict['opponent_attack_dict'] = {}                 
                
                for i, coordinate in enumerate(en_passant):
                        en_passant[i] = (7-coordinate[0], coordinate[1])                        
                for i, coordinate in enumerate(en_passant_loc): 
                        en_passant_loc[i] = (7-coordinate[0], coordinate[1])
                white_king_loc = (7-white_king_loc[0], white_king_loc[1])
                black_king_loc = (7-black_king_loc[0], black_king_loc[1])
                
                state_dict['white_king_loc'] = white_king_loc
                state_dict['black_king_loc'] = black_king_loc
                state_dict['en_passant'] = en_passant
                state_dict['en_passant_loc'] = en_passant_loc
                state_dict['current_move'] = current_move       
                state_dict['board'] = new_board         
                state_dict['white_value'] = white_value
                state_dict['black_value'] = black_value
                state_dict['value'] = value
                
                state = State(state_dict)
                self.states.append(state)
                self.state = self.states[-1]            
                return state
                
        def move_piece(self, move):
                if (self.state.turn == 'white'):
                        return(self.move_white_piece(move))
                else:
                        return(self.move_black_piece(move))
        
        def legal_moves(self):
                self.state.opponent_attack_dict = self.gen_check_attacks()
                
                for pseudo_move in self.gen_moves():                    
                        if (self.king_in_check(pseudo_move) == True):
                                continue
                                
                        if (pseudo_move == ((7, 4), (7, 2)) and self.state.board[7][4].upper() == 'K'):
                                if (self.king_in_check(((7, 4), (7, 4))) == True or self.king_in_check(((7, 4), (7, 3))) == True):
                                        continue                                        
                        elif (pseudo_move == ((7, 4), (7, 6)) and self.state.board[7][4].upper() == 'K'):
                                if (self.king_in_check(((7, 4), (7, 4))) == True or self.king_in_check(((7, 4), (7, 5))) == True):
                                        continue        
                        
                        yield(pseudo_move)
                        
        def move(self, move):
                coordinateMove = self.format_move(move)
                if (coordinateMove == False):
                        return(False)
                
                legal_move_list = list(self.state.legal_move_list)
                if (legal_move_list == []):
                        self.state.legal_move_list = self.legal_moves()
                        legal_move_list = list(self.state.legal_move_list)
                
                if (coordinateMove not in legal_move_list):
                        print('Not a valid move')
                        return(False)
                                
                self.move_piece(coordinateMove)
                
                checkmate = False
                self.state.legal_move_list = self.legal_moves()
                legal_move_list = list(self.state.legal_move_list)
                if (legal_move_list == []):
                        checkmate = True        

                check = self.king_ray_check()

                if (check == True and checkmate == True):
                        print('Checkmate!')
                        return(None)
                elif (check == True and checkmate == False):
                        print('Check!')
                elif (check == False and checkmate == True):
                        print('Stalemate!')
                        return(None)
                
                return(True)
        
        def undo_move(self):            
                self.states.pop()
                self.state = self.states[-1]
                return(True)            
        
        def print_board(self):
                print()
                for i, row in enumerate(self.state.board):
                        print(' ', 8-i, ' '.join(p for p in row))
                print('    h g f e d c b a \n\n')
                        
        def coordinate_to_notation(self, coordinate):
                rank_start = self.coordinate_to_notation_ranks[coordinate[0][0]]
                file_start = self.coordinate_to_notation_files[coordinate[0][1]]
                rank_end = self.coordinate_to_notation_ranks[coordinate[1][0]]
                file_end = self.coordinate_to_notation_files[coordinate[1][1]]
                
                notation = '{}{}{}{}'.format(file_start, rank_start, file_end, rank_end)
                return(notation)

        def notation_to_coordinate(self, notation):
                file_start = self.notation_to_coordinate_files[notation[0]]
                rank_start = self.notation_to_coordinate_ranks[int(notation[1])]
                file_end = self.notation_to_coordinate_files[notation[2]]
                rank_end = self.notation_to_coordinate_ranks[int(notation[3])]
                
                coordinate = ((rank_start, file_start), (rank_end, file_end))
                return(coordinate)
        
        def convert_move(self, move):
                if (len(move) != 4):
                        print('Input move in the form such as: e2e4')
                        return(False)
                
                if ((move[0] not in self.files) or (move[2] not in self.files)):
                        print('Input move in the form such as: e2e4')
                        return(False)
                        
                if ((int(move[1]) not in self.ranks) or (int(move[3]) not in self.ranks)):
                        print('Input move in the form such as: e2e4')
                        return(False)
                        
                if (self.state.turn == 'white'):
                        return(move)
                elif (self.state.turn == 'black'):
                        newMove = ''
                        newMove += move[0]
                        newMove += str(self.reverse_rank[int(move[1])])
                        newMove += move[2]
                        newMove += str(self.reverse_rank[int(move[3])])
                        return(newMove)
        def input_state(self, input_state_dict=None):
                if (input_state_dict == None):
                        state_dict = {}         
                        state_dict['board'] = [['r', 'k', '· ', '· ', '· ', '· ', '· ', '· '],
                                                                   ['· ', 'p', '· ', 'p', '· ', 'p', 'B', '· '],
                                                                   ['· ', 'P', '· ', '· ', '· ', '· ', '· ', '· '],
                                                                   ['p', 'N', 'p', '· ', '· ', '· ', 'P', '· '],
                                                                   ['P', '· ', '· ', '· ', '· ', '· ', '· ', '· '],
                                                                   ['· ', '· ', '· ', '· ', 'r', 'Q', 'K', '· '],
                                                                   ['· ', 'N', '· ', 'q', '· ', '· ', '· ', 'R'],
                                                                   ['· ', '· ', '· ', '· ', '· ', '· ', '· ', '· ']]
                                                                
                        state_dict['board']
                        state_dict['turn'] = 'white'
                        state_dict['current_move'] = None
                        state_dict['white_king_loc'] = (5, 6)
                        state_dict['black_king_loc'] = (0, 1)
                        state_dict['white_castle'] = [False, False]
                        state_dict['black_castle'] = [False, False]
                        state_dict['en_passant'] = []
                        state_dict['en_passant_loc'] = []
                        
                        state_dict['opponent_attack_dict'] = {}
                        state_dict['legal_move_list'] = {}
                        
                        #Value Hueristic
                        state_dict['white_value'] = 0
                        state_dict['black_value'] = 0
                        state_dict['value'] = 0
                
                        state = State(state_dict)
                        self.states.append(state)
                        self.state = self.states[-1]
                else:                   
                        state = State(input_state_dict)
                        self.states.append(state)
                        self.state = self.states[-1]
