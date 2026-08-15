class LinearRegression:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.x_stand = []
        self.y_stand = []
        self.x_mean = 0
        self.y_mean = 0
        self.x_std = 0
        self.y_std = 0
        self.w = 0
        self.b = 0
        self.w_grad = 0
        self.b_grad = 0

    @staticmethod
    def _standardize(l):
        mean = sum(l) / len(l)
        variance = sum([(x-mean)**2 for x in l]) / len(l)
        std = variance ** 0.5 #squareroot
        return [(x - mean ) / std for x in l] , std, mean

    def _transform(self):
        self.x_stand, self.x_std, self.x_mean = self._standardize(self.x)
        self.y_stand, self.y_std, self.y_mean = self._standardize(self.y)
