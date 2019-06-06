import numpy as np

def rand_graph(num_cities, percent_connected=.75):
    """
    Returns (symmetric) weight and connectivity matrixes.
    Weights are set to 0 when two nodes are not connected.
    Connectivity matrix has 1's when two nodes are connected
    and 0 otherwise.
    """
    
    weights = np.random.rand(num_cities, num_cities)
    weights = weights.T @ weights

    connected = np.ones((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(i):
            if np.random.rand(1) > percent_connected:
                connected[i, j] = 0
                connected[j, i] = 0
                weights[i, j] = 0
                weights[j, i] = 0
    for i in range(num_cities):
        connected[i, i] = 0
        weights[i, i] = 0

    return weights, connected

def classical(weight_matrix, connectivity_matrix, loop=True):
    """
    Returns optimal path and cost, if none exists then returns infinity and None.

    If loop is True, then the path will loop back to starting city.
    """

    num_cities = weight_matrix.shape[0]
    best_path = None
    best_cost = float("inf")

    for start_city in range(num_cities):
        cur_paths = []

        # initial connections from 0 to i
        for i in range(num_cities):
            if connectivity_matrix[start_city, i]:
                cur_paths.append(([start_city, i], weight_matrix[start_city, i]))

        for j in range(num_cities-2):
            new_paths = []
            for path, cost in cur_paths:
                cur_city = path[-1]
                for i in range(num_cities): # was range(1, num_cities)
                    if i not in path and connectivity_matrix[cur_city, i]:
                        new_paths.append((path + [i], cost + weight_matrix[cur_city, i]))
            cur_paths = new_paths

        # best_path = None
        # best_cost = float("inf")

        if loop:
            # loops back to start_city
            for path, cost in cur_paths:
                cur_city = path[-1]
                if connectivity_matrix[cur_city, start_city]:
                    new_cost = cost + weight_matrix[cur_city, start_city]
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_path = path + [start_city]
        else:
            # don't loop back to origin city
            for path, cost in cur_paths:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

    return best_cost, best_path


# takes a bitstring of 0's and 1's and returns a path
# e.g. [0, 1, 1, 0] -> [1, 0]
# returns None if invalid path
def bitstring_to_path(bitstring):
    num_cities = int(np.sqrt(len(bitstring)))

    path = []
    for i in range(num_cities):
        cur_city_str = bitstring[i*num_cities : (i+1)*num_cities]

        cur_city = -1
        for j in range(num_cities):
            if cur_city_str[j] == 1:
                if cur_city != -1:
                    return None
                cur_city = j
        if cur_city == -1:
            return None
        path.append(cur_city)

    return path

# path should be an array of cities either e.g. [0, 2, 3, 1]
# NOT IMPLEMENTED FOR LOOPING BACK TO START
# returns ininity if its an invalid path (not connected)
def calc_cost(path, weight_matrix, connectivity_matrix):
    cost = 0
    visited = set()
    for i in range(len(path) - 1):
        cur_city = path[i]
        next_city = path[i+1]
        if not connectivity_matrix[cur_city, next_city] or next_city in visited:
            return float("inf")
        visited.add(cur_city)
        cost += weight_matrix[cur_city, next_city]

    return cost



# example usage
# weights, connections = rand_graph(4)
# print(classical(weights, connections))

if __name__ == "__main__":

    NUM_CITIES = 4
    PERCENT_CONNECTED = .8

    weights, connections = rand_graph(NUM_CITIES, PERCENT_CONNECTED)

    print("Weight matrix")
    print(weights)
    print("-------------")
    print("Connectivity matrix")
    print(connections)
    print("-------------")
    print(classical(weights, connections, loop=False))
