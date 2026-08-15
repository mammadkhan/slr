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
        self.lr = 0.01

    @staticmethod
    def _standardize(l):
        mean = sum(l) / len(l)
        variance = sum([(x-mean)**2 for x in l]) / len(l)
        std = variance ** 0.5 #squareroot
        return [(x - mean ) / std for x in l] , std, mean

    def _transform(self):
        self.x_stand, self.x_std, self.x_mean = self._standardize(self.x)
        self.y_stand, self.y_std, self.y_mean = self._standardize(self.y)

    def predict(self,x):
        # f(x) = wx + b
        return self.w * x + self.b

    def cost(self):
        # 1/2m * sum((y_hat - y) ** 2)
        m = len(self.x_stand)
        cost_right_side = 0
        for i in range(m):
            cost_right_side += (self.predict(self.x_stand[i])-self.y_stand[i])**2
        return 1/(2*m) * cost_right_side

    def grad(self):
        # C(w,b) = 1/2m * sum((y_hat-y)**2) = 1/2m * sum(((wx+b)-y)**2)
        # dC/dw = (wx+b-y)**2 = 2(wx+b-y) * x -> 1/2m * sum(2x(wx+b-y)) = 1/m * sum(x(wx+b-y)) chain rule derv
        # dC/db = (wx+b-y)**2 = 2(wx+b-y) * 1 -> 1/2m * sum(2(wx+b-y)) = 1/m * sum(wx+b-y)
        m = len(self.x_stand)
        for i in range(m):
            self.w_grad += self.x_stand[i] * (self.predict(self.x_stand[i]) - self.y_stand[i])
            self.b_grad += self.predict(self.x_stand[i]) - self.y_stand[i]
        self.w_grad = self.w_grad * 1/m
        print(self.w_grad)
        self.b_grad = self.b_grad * 1/m

    def train(self,iter=100):
        self._transform()
        for i in range(iter):
            self.grad()
            self.w = self.w - self.lr * self.w_grad
            self.b = self.b - self.lr * self.b_grad
