#Tic-Tac-Toe with reinforcement learning / dynamic programming

from random import random
from math import log

#Make an array with each board state and the probability of winning from that state
#How many states are there? 
#I'll naively say 3^9 = 19683
#This is an upper bound, but does include some invalid states.
values = [0.5 for i in range(3**9)]
verbose = True

#Fills the probability of winning as 1 if the board state is a win and 0 if it's a loss.
#Also populates invalid board states with 0...
def populate_known():
	global values
	for ind in range(len(values)):
		board = index_state(ind)
		if not is_valid(board):
			values[ind] = 0.0 #Invalid state
		out = outcome(board)
		if out == 1:
			values[ind] = 1.0 #Winning state
		if out == -1:
			values[ind] = 0.0 #Losing state
		

#Gets the board state for the given index of the value grid
def index_state(index):
	board = [[0,0,0],[0,0,0],[0,0,0]]
	for i in range(3):
		for j in range(3):
			board[i][j] = (index//3**(3*i+j))%3
	return board

#Gets the index of the value grid for the given board state
def state_index(board):
	ind = 0
	for i in range(3):
		for j in range(3):
			ind += board[i][j]*3**(3*i+j)
	return ind

#Returns true if the board state is valid, false otherwise
def is_valid(board):
	count_x = 0
	count_o = 0
	for i in range(3):
		for j in range(3):
			if board[i][j] == 1:
				count_x += 1
			elif board[i][j] == 2:
				count_o += 1
	return (abs(count_x-count_o) <= 1)

#Returns 1 if win, -1 if loss, 0 otherwise
#Assumes board is valid
def outcome(board):
	linevals = []
	for i in range(3):
		linevals.append(board[i][0]*board[i][1]*board[i][2])
	for j in range(3):
		linevals.append(board[0][j]*board[1][j]*board[2][j])
	linevals.append(board[0][0]*board[1][1]*board[2][2])
	linevals.append(board[2][0]*board[1][1]*board[0][2])
	for k in range(8):
		if linevals[k] == 1:
			return 1
		if linevals[k] == 8:
			return -1
	return 0

#Must have already checked for win
def is_full(board):
	prod = 1
	for i in range(3):
		for j in range(3):
			prod *= board[i][j]
	return (prod != 0)

#Have player move randomly
def make_null_move(board):
	global verbose
	if verbose:
		print('Making random move...')
	ind = state_index(board)
	valid = False
	while not valid:
		pos = int(9*random())
		if board[pos//3][pos%3] == 0:
			board[pos//3][pos%3] = 1
			valid = True
	return board, ind

#Have opponent move randomly
def make_opp_move(board):
	global verbose
	if verbose:
		print('Opponent making move...')
	ind = state_index(board)
	valid = False
	while not valid:
		pos = int(9*random())
		if board[pos//3][pos%3] == 0:
			board[pos//3][pos%3] = 2
			valid = True
	return board, ind

def make_human_move(board):
	ind = state_index(board)
	valid = False
	while not valid:
		index = input('Enter position (A1, etc.): ')
		i = -1
		j = -1
		if index[0].lower() == 'a':
			j = 0
		elif index[0].lower() == 'b':
			j = 1
		elif index[0].lower() == 'c':
			j = 2
		else:
			print('Try again')
		if index[1] == '1':
			i = 0
		elif index[1] == '2':
			i = 1
		elif index[1] == '3':
			i = 2
		else:
			print('Try again')
		if i != -1 and j != -1:
			if board[i][j] == 0:
				board[i][j] = 2
				valid = True
			else:
				print('There\'s already something there...')
	return board, ind

#Make epsilon-greedy move with update rate alpha, using the state before the opponent's last move as reference
def make_move(board, epsilon, alpha, laststate, update=True):
	global values, verbose
	old_ind = state_index(board)
	ind = state_index(board)
	r = random()
	if r < epsilon:
		if verbose:
			print('Being explorative...')
		#Exploratory move
		valid = False
		while not valid:
			pos = int(9*random())
			if board[pos//3][pos%3] == 0:
				board[pos//3][pos%3] = 1
				ind += 3**pos
				valid = True
		#Don't update value function for exploratory move
	else:
		if verbose:
			print('Being greedy...')
		board, ind = make_move_greedy(board)
		#TODO: Tab this in one level
		#Update value function
		if update:
			values[laststate] += alpha*(values[ind] - values[laststate])
	return board, old_ind

#Have AI make move...
def make_move_greedy(board):
	global values, verbose
	ind = state_index(board)
	#Look at the values of all possible moves
	maxv = 0
	maxi = []
	maxj = []
	for i in range(3):
		for j in range(3):
			if board[i][j] == 0:
				val = values[ind + 3**(3*i+j)]
				if val > maxv:
					maxv = val
					maxi = [i]
					maxj = [j]
				elif val == maxv:
					maxi.append(i)
					maxj.append(j)
	if verbose:
		print('{0:.2f}% chance of winning with this move...'.format(maxv*100))
	#Random choice among best options
	choice = int(random()*len(maxi))
	board[maxi[choice]][maxj[choice]] = 1
	return board, ind + 3**(3*maxi[choice] + maxj[choice])

#Call when there is a loss, so no next move is available. Treat the next move as having exactly zero probability of winning.
#For wins, since you are making the final move, a similar update happens already.
def update_value_loss(board, alpha, laststate):
	global values
	ind = state_index(board)
	values[laststate] += alpha*(values[ind] - values[laststate])

def to_char(board, i, j):
	if board[i][j] == 0:
		return ' '
	elif board[i][j] == 1:
		return 'X'
	else:
		return 'O'

def print_board(board):
	print('{0}|{1}|{2}\n-+-+-\n{3}|{4}|{5}\n-+-+-\n{6}|{7}|{8}'.format(to_char(board,0,0), to_char(board,0,1), to_char(board,0,2), to_char(board,1,0), to_char(board,1,1), to_char(board,1,2), to_char(board,2,0), to_char(board,2,1), to_char(board,2,2)))

#Plays a game.
#player = true means the reinforcement learner goes first
#human = true has a human opponent
def play(player, epsilon, alpha, human=False):
	global verbose
	board = [[0,0,0],[0,0,0],[0,0,0]]
	ended = False
	turn = 0
	result = 0
	ind = 0
	while not ended:
		if player:
			if turn%2 == 0:
				board, _ = make_move(board, epsilon, alpha, ind)
			else:
				if human:
					board, ind = make_human_move(board)
				else:
					board, ind = make_opp_move(board)
		else:
			if turn%2 == 0:
				if human:
					board, ind = make_human_move(board)
				else:
					board, ind = make_opp_move(board)
			else:
				board, _ = make_move(board, epsilon, alpha, ind)
		if verbose:
			print_board(board)
		out = outcome(board)
		if out == 1:
			result = 1
			if verbose:
				if human:
					print('The AI won!')
				else:
					print('Player won!')
			ended = True
		elif out == -1:
			result = -1
			update_value_loss(board, alpha, ind)
			if verbose:
				if human:
					print('You won!')
				else:
					print('Opponent won!')
			ended = True
		elif is_full(board):
			if verbose:
				print('Draw!')
			ended = True
		turn += 1
	return result

def play_self(player, epsilon, alpha):
	global verbose
	board = [[0,0,0],[0,0,0],[0,0,0]]
	ended = False
	turn = 0
	result = 0
	ind = 0
	while not ended:
		if player:
			if turn%2 == 0:
				board, _ = make_move(board, epsilon, alpha, ind)
			else:
				board, ind = make_move(board, epsilon, alpha, ind, False)
		else:
			if turn%2 == 0:
				board, ind = make_move(board, epsilon, alpha, ind, False)
			else:
				board, _ = make_move(board, epsilon, alpha, ind)
		if verbose:
			print_board(board)
		out = outcome(board)
		if out == 1:
			result = 1
			if verbose:
				print('Player won!')
			ended = True
		elif out == -1:
			result = -1
			update_value_loss(board, alpha, ind)
			if verbose:
				print('Opponent won!')
			ended = True
		elif is_full(board):
			if verbose:
				print('Draw!')
			ended = True
		turn += 1
	return result

def play_null(player):
	global verbose
	board = [[0,0,0],[0,0,0],[0,0,0]]
	ended = False
	turn = 0
	result = 0
	ind = 0
	while not ended:
		if player:
			if turn%2 == 0:
				board, ind = make_null_move(board)
			else:
				board, ind = make_opp_move(board)
		else:
			if turn%2 == 0:
				board, ind = make_opp_move(board)
			else:
				board, ind = make_null_move(board)
		if verbose:
			print_board(board)
		out = outcome(board)
		if out == 1:
			result = 1
			if verbose:
				print('Player 1 won!')
			ended = True
		elif out == -1:
			result = -1
			if verbose:
				print('Player 2 won!')
			ended = True
		elif is_full(board):
			if verbose:
				print('Draw!')
			ended = True
		turn += 1
	return result

def estimate_null(n):
	wins = 0
	losses = 0
	for i in range(n):
		player = (random() < 0.5)
		out = play_null(player)
		if out == 1:
			wins += 1
		if out == -1:
			losses += 1
	return wins/n, losses/n

def estimate_win_chance(epsilon, alpha, n):
	wins = 0
	losses = 0
	for i in range(n):
		player = (random() < 0.5)
		out = play(player,epsilon, alpha, False)
		if out == 1:
			wins += 1
		if out == -1:
			losses += 1
	return wins/n, losses/n

def estimate_state_space():
	global values
	has_spoken = False
	count = 0
	for i in range(len(values)):
		if values[i] != 0.5:
			count += 1
		else:
			if random() < 0.1 and not has_spoken:
				#print('Example:')
				#print_board(index_state(i))
				has_spoken = True
	return count/len(values)

def run():
	global verbose
	#Estimate null model chance
	verbose = False
	print('Checking null model...')
	win_chance, loss_chance = estimate_null(10000)
	print('{0:.2f}% win, {1:.2f}% loss'.format(win_chance*100, loss_chance*100))
	input('(Press any key to continue)')
	print('Populating known values...')
	populate_known()
	print('Ready.')
	#state_space = estimate_state_space()
	#print('{:.2f}% of state space used'.format(state_space*100))
	epsilon = 0.001
	alpha = 1.0
	#Play 1000 games
	#Play 10k games?
	print('Playing...')
	gamesets = 200
	n = 1000
	for g in range(1,gamesets):
		alpha = 1 - g/gamesets #Decreasing the learning rate over time like this seems to really help.
		wins = 0
		losses = 0
		for k in range(n):
			player = (random() < 0.5)
			out = play(player, epsilon, alpha)
			if out == 1:
				wins += 1
			elif out == -1:
				losses += 1
		win_chance = wins/n
		loss_chance = losses/n
		print('{0:.2f}% win, {1:.2f}% loss'.format(win_chance*100, loss_chance*100))
	#for g in range(gamesets):
	#	win_chance, loss_chance = estimate_win_chance(epsilon, alpha, n)
	#	print('{0:.2f}% win, {1:.2f}% loss'.format(win_chance*100, loss_chance*100))
	#	state_space = estimate_state_space()
	#	print('{:.2f}% of state space used'.format(state_space*100))	
	#games = 10000
	#for g in range(games):
	#	player = (random() < 0.5)
	#	play(player, epsilon, alpha)
	#	if g % 10 == 0:
	#		print('{0:.2f}% done...'.format(100*g/games))
	#Now show output
	print('Trained.')
	verbose = True
	inp = input()
	while inp != 'stop':
		player = (random() < 0.5)
		play(player, epsilon, alpha, inp.lower() == 'me')
		inp = input('(Enter "me" to play)')
	

if __name__=='__main__':
	run()