import numpy as np
import matplotlib.pyplot as plt
from sko.PSO import PSO

# Sample observed data
observed_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Sample model function: You will need to replace this with your actual model
def model(params, x):
    a, b, c = params
    return a * x ** 2 + b * x + c

# Objective function: RMSE between model output and observed data
def objective_function(params):
    x = np.arange(1, len(observed_data) + 1)
    predictions = model(params, x)
    rmse = np.sqrt(np.mean((predictions - observed_data) ** 2))
    return rmse

# PSO parameters
num_particles = 30
dimensions = 3  # Number of parameters to optimize
max_iter = 100

# Initialize PSO
pso = PSO(func=objective_function, dim=dimensions, pop=num_particles, max_iter=max_iter, lb=[-10, -10, -10], ub=[10, 10, 10])

# Run PSO
pso.run()

# Output the results
best_params = pso.gbest_x
best_rmse = pso.gbest_y

print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)

# Plot the convergence
plt.plot(pso.gbest_y_hist)
plt.xlabel('Iteration')
plt.ylabel('Best RMSE')
plt.title('Convergence of PSO')
plt.show()
