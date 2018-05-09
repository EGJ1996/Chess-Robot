from chess import Chess
import random
import collections
import time
class MinimaxGameTree():
	def __init__(self, chess, color, depth):
		self.chess = chess
		self.player = color
		self.depth = depth
		self.fulltime = 0
		
	def iterativeDeepening(self):
		alpha = -40000
		beta = 40000
		pv = []		
		for depth in range(1, self.depth+1):
			data = self.dfsMax(alpha, beta, depth, pv)
			pv = data[1]
		
		best_value = data[0]
		move_list = data[1]
		best_move = move_list[self.depth-1]
		return(best_move)
	
	def dfsMax(self, alpha, beta, depth, pv):
	
		if (depth == 0):
			value = self.evaluate_board(self.player)
			return((value, []))
		
		move_list = []
		best_move = None
		if (pv != []):
			move = pv.pop()
			self.next_position(move)			
			data = self.dfsMin(alpha, beta, depth-1, pv)
			self.previous_position()			
			value = data[0]
			
			if (value >= beta):				
				move_list = data[1]
				move_list.append(best_move)
				return((beta, move_list))			
			if (value > alpha):					
				alpha = value
				best_move = move
				move_list = data[1]
		
		for move in self.chess.legal_moves():
			self.next_position(move)			
			data = self.dfsMin(alpha, beta, depth-1, pv)
			self.previous_position()			
			value = data[0]
			
			if (value >= beta):				
				move_list = data[1]
				move_list.append(best_move)
				return((beta, move_list))			
			if (value > alpha):					
				alpha = value
				best_move = move
				move_list = data[1]
					
		if (best_move == None):
			alpha = -20000
		
		move_list.append(best_move)
		return((alpha, move_list))
	
	def dfsMin(self, alpha, beta, depth, pv):
	
		if (depth == 0):
			value = self.evaluate_board(self.player)
			return((value, []))
		
		move_list = []
		best_move = None
		if (pv != []):
			move = pv.pop()
			self.next_position(move)			
			data = self.dfsMax(alpha, beta, depth-1, pv)
			self.previous_position()
			value = data[0]
			
			if (value <= alpha):
				move_list = data[1]
				move_list.append(best_move)
				return((alpha, move_list))			
			if (value < beta):
				beta = value
				best_move = move
				move_list = data[1]
		
		for move in self.chess.legal_moves():
			self.next_position(move)			
			data = self.dfsMax(alpha, beta, depth-1, pv)
			self.previous_position()			
			value = data[0]
			
			if (value <= alpha):
				move_list = data[1]
				move_list.append(best_move)
				return((alpha, move_list))			
			if (value < beta):
				beta = value
				best_move = move
				move_list = data[1]
					
		if (best_move == None):
			beta = 20000
		
		move_list.append(best_move)
		return((beta, move_list))
		
	def evaluate_board(self, color):	
		if (color == 'white'):
			value = self.chess.state.value 
		if (color == 'black'):
			value = -self.chess.state.value
		
		return(value)
		
	def next_position(self, move):			
		self.chess.move_piece(move)
	
	def previous_position(self):
		self.chess.undo_move()


class Algorithm():
	
	def __init__(self, chess, player, depth):
		self.chess = chess
		self.player = player
		self.depth = depth
		self.fulltime = 0
		
	def best_move(self):	
		self.tree = MinimaxGameTree(self.chess, self.player, self.depth)
		move = self.tree.iterativeDeepening()
		notation = self.chess.coordinate_to_notation(move)
		return(notation)
