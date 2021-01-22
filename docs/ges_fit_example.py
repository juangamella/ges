import ges
import ges.scores
import sempler
import numpy as np

# Generate observational data from a Gaussian SCM using sempler
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape) # sample weights
data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)

# Define the score class
score_class = ges.scores.GaussObsL0Pen(data)

# Run GES with the gaussian BIC score
estimate, score = ges.fit(score_class)

print(estimate, score)

# Output
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 1 0]] 24002.112921580803
