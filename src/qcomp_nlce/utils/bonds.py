import numpy as np
from itertools import product


def bond_coordinates(m, n):
    """
    Create a 2D array of coordinate tuples (i, j) for an m x n grid.
    Indices go from 0 to m-1 and 0 to n-1.
    Returns a flat list of tuples in row-major order (matching Julia's column-major reshape).
    """
    indices = [(i, j) for j in range(n) for i in range(m)]
    return indices


def open_bonds(N):
    """
    Create bonds for an open (non-periodic) 1D chain of N sites.
    Returns list of [i, i+1] pairs for i = 1 to N-1 (1-indexed).
    """
    bonds = [[i, i + 1] for i in range(1, N)]
    return bonds


def periodic_bonds(N):
    """
    Create bonds for a periodic 1D chain of N sites.
    Includes all open bonds plus the wrap-around bond [1, N] (1-indexed).
    """
    per_link = [[1, N]]
    periodic = open_bonds(N) + per_link
    return periodic


def graph(m, n):
    """
    Create bonds for an m x n 2D lattice with open boundary conditions.
    Two sites are connected if their Euclidean distance is 1 (nearest neighbors).
    Returns 1-indexed bonds.
    """
    coordinates = bond_coordinates(m, n)

    bonds = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            # Calculate Euclidean distance between coordinates
            diff = np.array(coordinates[i]) - np.array(coordinates[j])
            if np.linalg.norm(diff) == 1:
                # Julia is 1-indexed, so we add 1 to match
                bonds.append([i + 1, j + 1])

    return bonds


def graph_periodic_1d(m, n):
    """
    Create bonds for an m x n 2D lattice with periodic boundary in one direction.
    Same as graph() but adds an extra bond connecting the last site to the first.
    Returns 1-indexed bonds.
    """
    coordinates = bond_coordinates(m, n)

    bonds = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            diff = np.array(coordinates[i]) - np.array(coordinates[j])
            if np.linalg.norm(diff) == 1:
                bonds.append([i + 1, j + 1])

    # Add periodic bond connecting last site to first (1-indexed)
    bonds.append([m * n, 1])

    return bonds


if __name__ == "__main__":
    # Test examples
    print("Open bonds (N=5):", open_bonds(5))
    print("Periodic bonds (N=5):", periodic_bonds(5))
    print("Graph 2x3:", graph(2, 3))
    print("Graph periodic 1D (2x3):", graph_periodic_1d(2, 3))
