class LocalSearchSwitch:

    def __init__(self, warehouses, customers):
        """
        Initializes an instance of the ClassName class.

        Args:
            warehouses (list): A list of warehouses.
            customers (list): A list of customers.

        Attributes:
            warehouses (list): A list of warehouses.
            customers (list): A list of customers.
            current_solution (list): The current solution.
            best_solution (list): The best solution.
            best_cost (float): The cost of the best solution.
        """
        self.warehouses = warehouses
        self.customers = customers
        self.current_solution = [i % 2 == 0 for i in range(len(warehouses))]
        self.best_solution = self.current_solution[:]
        self.best_cost = self.calculate_cost(self.current_solution)
    
    def calculate_cost(self, solution):
        """
        Calculates the total cost of a given solution.

        Args:
            solution (list): A list representing the solution, where each element indicates whether a facility is open or not.

        Returns:
            float: The total cost of the solution.
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
            total_cost += min_cost
        return total_cost
    

    def generate_neighbors(self):
        """
        Generates neighboring solutions by flipping the value of each warehouse in the current solution.

        Returns:
            list: A list of neighboring solutions.
        """
        neighbors = []
        for i in range(len(self.warehouses)):
            neighbor_solution = self.current_solution[:]
            neighbor_solution[i] = not neighbor_solution[i]  
            neighbors.append(neighbor_solution)
        return neighbors

    def local_search(self):
        """
        Performs a local search to find an improved solution.

        This method iteratively generates neighboring solutions and checks if they have a lower cost than the current best solution.
        If a better solution is found, it updates the best solution and its cost.
        The process continues until no further improvement is made.

        Returns:
            tuple: A tuple containing the best solution and its cost.
        """
        improvement = True
        while improvement:
            improvement = False
            neighbors = self.generate_neighbors()
            for neighbor in neighbors:
                neighbor_cost = self.calculate_cost(neighbor)
                if neighbor_cost < self.best_cost:
                    self.best_solution = neighbor
                    self.best_cost = neighbor_cost
                    improvement = True
            self.current_solution = self.best_solution[:]
        return self.best_solution, self.best_cost