import numpy as np
import math
import sys

Q = ('state A', 'state B')

# read hmm file function
def read_hmm(filename):
    with open(filename) as f:
        HMM_file = f.read().split()
        # read states and symbols
        num_states = int(HMM_file[0])
        num_symbols = int(HMM_file[1])

        # initial Transition_matrix A and Emission_matrix E
        A = np.zeros((num_states, num_states))
        E = np.zeros((num_states, num_symbols))

        # read Transition_matrix A and Emission_matrix E
        x = num_states + 3
        for i in range(0, num_states):
            for j in range(num_states + num_symbols):
                if j < num_states:
                    A[i][j] = float(HMM_file[x])
                    x+=1
                else:
                    E[i][j - num_states] = float(HMM_file[x])
                    x+=1

        # read initial probabilities
        PI = np.zeros((num_states, 1))
        for i in range(num_states):
            PI[i][0] = float(HMM_file[i + 3])

    f.close()
    return A, E, PI

# read faste file function
def read_fasta(filename):
    with open(filename) as f:
        file = f.readlines()
        # format observation sequence
        myfile = ''.join(file[1:])
        seq_upper = myfile.upper()
        seq = seq_upper.replace('A','0').replace('C','1').replace('G','2').replace('T','3').replace(' ','').replace('\n','').replace('\r','')
        ls = list(map(int, seq))
    f.close()
    return ls

# viterbi algorithm
def Viterbi(A, E, PI, Q, obs):      # A: Transition matrix  E: emission matrix  PI: intial probability  Q: states   obs: observation sequence
    num_states = len(Q)      # states number
    len_seq = len(obs)    # length of observation sequence
    V = np.array([[0] * num_states] * len_seq, dtype = np.float64)         # V: path probabilities table
    path = np.array([[0] * num_states] * len_seq, dtype = np.int64)         # P: hidden states path

    # initial states (t = 0)
    for i in range(0, num_states):
        V[0][i] = math.log(PI[i][0], 2) + math.log(E[i][obs[0]], 2)
        path[0][i] = 0

    # main of Viterbi (t > 0)
    for i in range(1, len_seq):
        for j in range(num_states):
            tmp = [V[i - 1, k] + math.log(A[k][j], 2) for k in range(num_states)]    # s(i-1) x Transition prob
            V[i][j] = max(tmp) + math.log(E[j][obs[i]], 2)                   # s(i) = Emission prob X max{s(i - 1) x Transition prob}
            path[i][j] = tmp.index(max(tmp))                                 # record path

    # last prob and path
    P = max(V[len_seq - 1, :])
    I = int(np.argmax(V[len_seq - 1, :]))

    # traceback part
    newpath = [I]
    for i in reversed(range(1, len_seq)):
        end = newpath[-1]
        newpath.append(path[i][end])

    hidden_states = [Q[i] for i in reversed(newpath)]

    return P, hidden_states

# output function
def output_segments(hidden_states):
    # output list
    index = 1
    list_hidden = list(hidden_states)
    num_stateB = []
    for i in range(1, len(list_hidden)):
        if list_hidden[i] is not list_hidden[i - 1]:
            print('%d %d %s' %(index, i, list_hidden[i - 1]))
            num_stateB.append(list_hidden[i - 1])
            index = i + 1
        else:
            if i == len(list_hidden) - 1:
                print("%d %d %s" %(index, i+1, list_hidden[i]))
                num_stateB.append(list_hidden[i])
            else:
                continue

    # output number of stateB
    key = 0
    for i in range(len(num_stateB)):
        if num_stateB[i] is Q[1]:
            key += 1
    print ('num of %s is %d'  %(Q[1],key))


def main():
    # receive input
    hmm_filename = sys.argv[1]
    fasta_filename = sys.argv[2]
    # reading file
    A, E, PI = read_hmm(hmm_filename)
    obs = read_fasta(fasta_filename)
    # run Viterbi
    P, hidden_states = Viterbi(A, E, PI, Q, obs)
    # output
    output_segments(hidden_states)

if __name__ == '__main__':
    main()