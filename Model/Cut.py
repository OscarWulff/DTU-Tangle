   class Cut:
    def __init__(self, id : int, cost : float, oreientation : str, A : set, Ac : set ):
        self.id = id
        self.cost = cost
        self.oreientation = oreientation
        self.A = A
        self.Ac = Ac
        

    def is_in(self, v):
        return v in self.A

    
    
   