class greedy:

    def __init__(self, warehouses, customers):
        """
        Initializes a Greedy object with the given warehouses and customers.

        Args:
            warehouses (list): A list of warehouses.
            customers (list): A list of customers.

        Attributes:
            warehouses (list): A list of warehouses.
            customers (list): A list of customers.
            current_solution (list): The current solution, represented as a list of booleans.
            best_solution (list): The best solution found so far, represented as a list of booleans.
            best_cost (float): The cost of the best solution found so far.
        """
        self.warehouses = warehouses
        self.customers = customers
        self.current_solution = [True for _ in range(len(warehouses))]
        self.best_solution = self.current_solution[:]
        self.best_cost = self.calculate_cost(self.current_solution)

   
    def greedy(self):
        """
        Applies the greedy algorithm to find the best solution for the UFLP problem.

        Returns:
            tuple: A tuple containing the best solution and its cost.
        """
        for i in range(len(self.warehouses)):
            self.current_solution[i] = False

            current_cost = self.calculate_cost(self.current_solution)
            if current_cost < self.best_cost:
                self.best_solution = self.current_solution[:]
                self.best_cost = current_cost
            else:
                self.current_solution[i] = True
    
        return self.best_solution, self.best_cost

    
    def calculate_cost(self, solution):
        """
        Calculates the total cost of a given solution for the Uncapacitated Facility Location Problem (UFLP).

        Parameters:
            solution (list): A list representing the solution, where each element indicates whether a facility is open or closed.

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
            if min_cost == float('inf'):
                return float('inf')
            total_cost += min_cost
        return total_cost