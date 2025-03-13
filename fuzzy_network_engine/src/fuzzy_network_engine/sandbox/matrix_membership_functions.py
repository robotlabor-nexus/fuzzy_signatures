import numpy as np

import matplotlib.pyplot as plt

def main():
    t = np.linspace(0, 1, 100)
    A = np.array([1, -0.5])
    print(np.dot(A,t))
    plt.show()

if __name__=="__main__":
    main()