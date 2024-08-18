import numpy as np

# Define the parameters
SearchAgents_no = 30  # Number of search agents (whales)
Max_iter = 500  # Maximum number of iterations
dim = 2  # Dimensionality of the search space
lb = -10 * np.ones(dim)  # Lower bound of the search space
ub = 10 * np.ones(dim)  # Upper bound of the search space

# Initialize the positions of search agents
Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb

# Initialize the leader (best solution found)
Leader_position = np.zeros(dim)
Leader_score = np.inf  # Change to -np.inf for maximization problems

# Objective function (RMSE)
def RMSE_ObjectiveFunction(x):
    # Example objective function: RMSE of a quadratic function
    # You can replace this with your actual objective function
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = x[0] * (np.arange(1, 6) ** 2) + x[1]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# Main loop of the WOA
for t in range(Max_iter):
    for i in range(SearchAgents_no):
        # Ensure search agents stay within the search space
        Positions[i, :] = np.clip(Positions[i, :], lb, ub)
        
        # Calculate the objective function (RMSE)
        fitness = RMSE_ObjectiveFunction(Positions[i, :])
        
        # Update the leader
        if fitness < Leader_score:  # Change to > for maximization problems
            Leader_score = fitness
            Leader_position = Positions[i, :].copy()
    
    # Update the position of search agents
    a = 2 - t * (2 / Max_iter)  # a decreases linearly from 2 to 0
    b = 1  # Define the constant b
    for i in range(SearchAgents_no):
        r1 = np.random.rand()  # r1 is a random number in [0, 1]
        r2 = np.random.rand()  # r2 is a random number in [0, 1]
        A = 2 * a * r1 - a  # Equation (2.3) in the paper
        C = 2 * r2  # Equation (2.4) in the paper
        
        p = np.random.rand()  # p in [0, 1]
        l = -1 + np.random.rand() * 2  # l is a random number in [-1, 1]
        
        if p < 0.5:
            if abs(A) >= 1:
                rand_leader_index = np.random.randint(0, SearchAgents_no)
                X_rand = Positions[rand_leader_index, :]
                D_X_rand = abs(C * X_rand - Positions[i, :])  # Equation (2.7)
                Positions[i, :] = X_rand - A * D_X_rand  # Equation (2.8)
            elif abs(A) < 1:
                D_Leader = abs(C * Leader_position - Positions[i, :])  # Equation (2.1)
                Positions[i, :] = Leader_position - A * D_Leader  # Equation (2.2)
        elif p >= 0.5:
            distance2Leader = abs(Leader_position - Positions[i, :])
            Positions[i, :] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + Leader_position  # Equation (2.5)
    
    # Display the best score found so far
    print(f"Iteration {t+1}: Best RMSE = {Leader_score}")

print(f"Best solution found: {Leader_position}")
print(f"Best RMSE: {Leader_score}")
