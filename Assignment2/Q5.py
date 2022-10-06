import numpy as np

# There are 25 variables, namely the state values of each of the cell in the 5x5 grid
# We also have 25 bellman eqautions, one for each state, with the same 25 varibles
# We will solve this using matrices by solving:
# Ax = B ==> x = inv(A)B

# Let the top left cell be [0, 0] and the bottom right cell be [4, 4]
# Let the cell in the cth column and rth row be numbered as 5 * r + c
# The special cases then are:
#       1 --> 21 (reward +10)
#       3 --> 13 (reward +5)

n = 25
gamma = 0.90
A = np.zeros([n, n], dtype='f')
B = np.zeros([25, 1], dtype='f')

for i in range(25):
    ## We add coefficients to the corresponding equation, refering to each of the four actions
    A[i][i] = 1.0
    if i == 1:
        ## special case 1
        B[i][0] += 10.0
        A[i][21] += -gamma
        
    elif i == 3:
        ## special case 2
        B[i][0] += 5.0
        A[i][13] += -gamma

    else:
        ## left
        if (i % 5) == 0:
            ## can't go left
            B[i][0] += (-0.25)
            A[i][i] += gamma * (-0.25)
        else:
            ## go left
            A[i][i - 1] += gamma * (-0.25)

        ## right
        if ((i + 1) % 5) == 0:
            ## can't go right
            B[i][0] += (-0.25)
            A[i][i] += gamma * (-0.25)
        else:
            ## go right
            A[i][i + 1] += gamma * (-0.25)
        
        ## up
        if i < 5:
            ## can't go up
            B[i][0] += (-0.25)
            A[i][i] += gamma * (-0.25)
        else:
            ## go up
            A[i][i - 5] += gamma * (-0.25)

        ## down
        if i > 19:
            ## can't go down
            B[i][0] += (-0.25)
            A[i][i] += gamma * (-0.25)
        else:
            ## go down
            A[i][i + 5] += gamma * (-0.25)


v = np.dot(np.linalg.inv(A), B).reshape([5, 5])
print(np.round(v, 1))