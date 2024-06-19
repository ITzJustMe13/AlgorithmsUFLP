from modules.warehouse import Warehouse
from modules.customer import Customer
from modules.searchSwitch import LocalSearchSwitch
from modules.searchSwap import LocalSearchSwap
from modules.greedy import greedy
from modules.geneticAlgorithm import GeneticAlgorithm
from modules.geneticAlgorithmWOLocalSearch import GeneticAlgorithmNoSearch
import time

if __name__ == '__main__':
    """
    This script reads data from a file, performs various optimization algorithms on the data,
    and prints the best solution and its cost.

    The script uses the following modules:
    - Warehouse: Represents a warehouse.
    - Customer: Represents a customer.
    - LocalSearchSwitch: Performs local search using switch operations.
    - LocalSearchSwap: Performs local search using swap operations.
    - Greedy: Implements a greedy algorithm.
    - GeneticAlgorithm: Implements a genetic algorithm.
    - GeneticAlgorithmNoSearch: Implements a genetic algorithm without local search.

    The script reads data from the file './data/ORLIB/ORLIB-uncap/70/cap71.txt' in the following format:
    - The first line contains the total number of warehouses and total number of customers.
    - The next lines contain information about each warehouse and customer, including costs.

    The script then runs different optimization algorithms and prints the best solution and its cost.

    Note: Uncomment the desired algorithm to run it.
    """
    with open('./data/M/Kcapmr5.txt', 'r') as file:
        total_warehouses, total_customers = map(int, file.readline().split())
        warehouses = [Warehouse((file.readline().split()[1]).rstrip('.')) for _ in range(total_warehouses)]
        customers: list[Customer] = []
        while True:
            line = file.readline()
            if not line:
                break
            elif len(line.split()) == 1:
                customers.append(Customer())
            else:
                customers[-1].costs.extend(line.split())
        if len(warehouses) != total_warehouses or len(customers) != total_customers:
            raise ValueError('Missing values')

    Greedy = greedy(warehouses, customers)
    GenAlg = GeneticAlgorithm(warehouses, customers)
    SearchSwitch = LocalSearchSwitch(warehouses, customers)
    SearchSwap = LocalSearchSwap(warehouses, customers)
    GenAlgNOSearch = GeneticAlgorithmNoSearch(warehouses, customers)

    start_time = time.time()

    #best_solution, best_cost = GenAlg.run()         ##q incrivelmente lento
    best_solution, best_cost = GenAlgNOSearch.run() ## incrivel rapido
    #best_solution, best_cost = Greedy.greedy()              ##BOM
    #best_solution, best_cost = SearchSwitch.local_search()    ##MT BOM
    #best_solution, best_cost = SearchSwap.local_search()    ##N PRESTA

    for i , warehouse in enumerate(warehouses):
        warehouse.is_open = best_solution[i]

    print("Melhor Soluçao: ", best_solution)
    print("Custo da melhor solução: ", best_cost)
    print("Tempo: %s segundos" % (time.time() - start_time))
