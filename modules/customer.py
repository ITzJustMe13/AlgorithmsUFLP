class Customer:
    def __init__(self) -> None:
        self.costs = []
        
    def __str__(self) -> str:
        return f'(Costs: {self.costs})'
    
    def __repr__(self) -> str:
        return self.__str__()