class Customer:
    def __init__(self) -> None:
        """
        Initializes a new instance of the Customer class.
        """
        self.costs = []
        
    def __str__(self) -> str:
        """
        Returns a string representation of the Customer object.
            
        Returns:
            str: A string representation of the Customer object, including the costs.
        """
        return f'(Costs: {self.costs})'
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Customer object.
        
        :return: A string representation of the Customer object.
        :rtype: str
        """
        return self.__str__()