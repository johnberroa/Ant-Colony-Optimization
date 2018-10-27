# Ant Colony Optimization
A Python implementation of the Ant Colony Optimization algorithm for generating solutions to such tasks as the Traveling Salesman Problem


## Example Usage
```python
problem = some_distance_matrix
optimizer = AntColonyOptimizer(ants=10, evaporation_rate=.1, intensification=2, alpha=1, beta=1,
                               beta_evaporation_rate=0, choose_best=.1)
 
best = optimizer.fit(problem, 100)
optimizer.plot()
```

## Example Plot
![ACO Fitted](ACO.png?raw=true "ACO Fitted")

###### Now 20x faster than my first attempt!
