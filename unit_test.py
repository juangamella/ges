import numpy as np
import ges

if __name__ == "__main__":
    x = np.random.randint(0, 5, 10000)
    y = (x + np.random.randint(0, 8, 10000)) % 8
    z = (x + np.random.randint(0, 8, 10000)) % 8

    data = np.stack([x, y, z], axis=-1)

    estimate, score = ges.fit_nml(data)
    print(estimate, score)
