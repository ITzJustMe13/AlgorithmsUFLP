class greedy:

    def __init__(self, warehouses, customers):
        self.warehouses = warehouses
        self.customers = customers
        self.current_solution = [True for _ in range(len(warehouses))]
        self.best_solution = self.current_solution[:]
        self.best_cost = self.calculate_cost(self.current_solution)

    #Algoritmo greedy
    def greedy(self):
        for i in range(len(self.warehouses)):
            self.current_solution[i] = False

            current_cost = self.calculate_cost(self.current_solution)
            if current_cost < self.best_cost:
                self.best_solution = self.current_solution[:]
                self.best_cost = current_cost
            else:
                self.current_solution[i] = True
    
        return self.best_solution, self.best_cost

    #calcula os custos
    def calculate_cost(self, solution):
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