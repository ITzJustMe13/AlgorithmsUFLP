import random
import multiprocessing

class LocalSearchSwap:

    def __init__(self, warehouses, customers):
        """
        Initializes a SearchSwap object with the given warehouses and customers.

        Parameters:
        - warehouses (list): A list of warehouses.
        - customers (list): A list of customers.

        Attributes:
        - self.warehouses (list): The list of warehouses.
        - self.customers (list): The list of customers.
        - self.current_solution (list): The current solution, represented as a list of boolean values.
        - self.best_solution (list): The best solution found so far, represented as a list of boolean values.
        - self.best_cost (float): The cost of the best solution found so far.
        """
        self.warehouses = warehouses
        self.customers = customers
        self.current_solution = [random.choice([True, False]) for _ in range(len(warehouses))]
        if not any(self.current_solution):
            self.current_solution[random.randint(0, len(warehouses) - 1)] = True
        self.best_solution = self.current_solution[:]
        self.best_cost = self.calculate_cost(self.current_solution)
    
    
    def calculate_cost(self, solution):
        """
        Calculates the total cost of a given solution for the Uncapacitated Facility Location Problem (UFLP).

        Parameters:
        - solution (list): A binary list representing the solution, where each element indicates whether a facility is open or closed.

        Returns:
        - total_cost (float): The total cost of the solution.

        """
        total_cost = 0
        for i, facility_open in enumerate(solution):
            if facility_open:
                total_cost += float(self.warehouses[i].fixed_cost)

        for customer in self.customers:
            min_cost = float('inf')
            for i, facility_open in enumerate(solution):
                if facility_open:
                    min_cost = min(min_cost, float(customer.costs[i]))
            if min_cost == float('inf'):
                return float('inf')
            total_cost += min_cost
        return total_cost
    

    def generate_neighbors(self, max_neighbors=10):
        """
        Generates a list of neighboring solutions by swapping open and closed indices.

        Args:
            max_neighbors (int): The maximum number of neighbors to generate. Defaults to 10.

        Returns:
            list: A list of neighboring solutions, where each solution is represented as a list of open and closed indices.
        """
        neighbors = []
        open_indices = [i for i, open_state in enumerate(self.current_solution) if open_state]
        closed_indices = [i for i, open_state in enumerate(self.current_solution) if not open_state]
        
        for _ in range(min(max_neighbors, len(open_indices) * len(closed_indices))):
            open_idx = random.choice(open_indices)
            closed_idx = random.choice(closed_indices)
            neighbor_solution = self.current_solution[:]
            neighbor_solution[open_idx], neighbor_solution[closed_idx] = (
                neighbor_solution[closed_idx], neighbor_solution[open_idx]
            )
            neighbors.append(neighbor_solution)
        
        return neighbors


    def evaluate_neighbor(self, neighbor):
        """
        Evaluates the cost of a given neighbor.

        Parameters:
        neighbor (object): The neighbor to be evaluated.

        Returns:
        float: The cost of the neighbor.
        """
        return self.calculate_cost(neighbor)

    def quick_evaluate(self, solution):
        """
        Calculates the evaluation score for a given solution.

        Parameters:
        solution (list): A list representing the solution.

        Returns:
        int: The evaluation score of the solution.
        """
        return sum(solution)
    
    
    def is_valid_solution(self, solution):
        """
        Checks if a given solution is valid.

        Args:
            solution (list): The solution to be checked.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        return any(solution)


    def local_search(self, max_iterations_without_improvement=10):
        """
        Perform local search to find the best solution for the given problem.

        Args:
            max_iterations_without_improvement (int): The maximum number of iterations without improvement allowed.

        Returns:
            tuple: A tuple containing the best solution found and its cost.

        """
        improvement = True
        iterations_without_improvement = 0
        
        while improvement and iterations_without_improvement < max_iterations_without_improvement:
            improvement = False
            neighbors = self.generate_neighbors()
            
            
            quick_scores = [self.quick_evaluate(neighbor) for neighbor in neighbors]
            
            
            sorted_neighbors = sorted(zip(quick_scores, neighbors), key=lambda x: x[0])
            
           
            with multiprocessing.Pool() as pool:
                detailed_scores = pool.map(self.evaluate_neighbor, [neighbor for _, neighbor in sorted_neighbors[:10]])
            
            for i, neighbor_cost in enumerate(detailed_scores):
                if neighbor_cost < self.best_cost:
                    self.best_solution = sorted_neighbors[i][1]
                    self.best_cost = neighbor_cost
                    improvement = True
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            
            self.current_solution = self.best_solution[:]
        
        return self.best_solution, self.best_cost