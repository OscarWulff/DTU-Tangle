class Cut:
    def __init__(self, id : int = None, cost : float = None, orientation : str = None, A : set = set(), Ac : set = set()):
        self.id = id
        self.cost : float = cost
        self.orientation = orientation
        self.A = A
        self.Ac = Ac
        self.line_placement = None
        self.A_points = []
        self.Ac_points = []

    def init(self):
        self.A = set()
        self.Ac = set() 
        self.A_points = []
        self.Ac_points = []
   