import numpy as np
from scipy.ndimage import label


class CellularAutomaton:
    def __init__(self, M, N, rule, p_alive=0.5, boundary="wrap"):
        self.M = M
        self.N = N
        self.rule = rule
        self.boundary = boundary

        self.grid = np.random.choice([0, 1], size=(M, N), p=[1 - p_alive, p_alive])

    def invert(self):
        for i in range(self.M):
            for j in range(self.N):
                self.grid[i, j] = 0 if self.grid[i, j] == 1 else 1

    def get_neighbors(self, i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di, j + dj

                if self.boundary == "wrap":
                    ni %= self.M
                    nj %= self.N
                    neighbors.append(self.grid[ni, nj])
                else:
                    if 0 <= ni < self.M and 0 <= nj < self.N:
                        neighbors.append(self.grid[ni, nj])
                    else:
                        neighbors.append(0)

        return neighbors

    def step(self):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.M):
            for j in range(self.N):
                state = self.grid[i, j]
                neighbors = self.get_neighbors(i, j)
                new_grid[i, j] = self.rule(state, neighbors)

        self.grid = new_grid

    def run(self, steps, remove_pockets=False):
        for _ in range(steps):
            self.step()
            if remove_pockets and _ == steps - 1:
                self.keep_largest_alive_component()

    def keep_largest_alive_component(self):
        """
        Keeps only the largest connected component of alive cells (1s).
        All other alive components are removed (set to 0).
        """
        labeled_array, num_features = label(self.grid)
        if num_features == 0:
            return None
        counts = np.bincount(labeled_array.ravel())
        largest_label = counts[1:].argmax() + 1
        result = (labeled_array == largest_label).astype(int)
        self.grid = result
