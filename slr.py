class LinearRegression:
    def __init__(self,x,y,lr=0.01,train_perc=0.8):
        self.x = x
        self.y = y
        self.lr = lr
        self.train_perc = train_perc
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.x_train_stand = []
        self.y_train_stand = []
        self.x_test_stand = []
        self.y_test_stand = []
        self.x_mean = 0
        self.y_mean = 0
        self.x_std = 0
        self.y_std = 0
        self.w = 0
        self.b = 0
        self.w_grad = 0
        self.b_grad = 0

    #training and testing data split without random
    def _train_test(self):
        divider = round(len(self.x)*self.train_perc)
        self.x_train,self.y_train = self.x[:divider],self.y[:divider]
        self.x_test,self.y_test = self.x[divider:],self.y[divider:]

    @staticmethod
    def _standardize(l):
        mean = sum(l) / len(l)
        variance = sum([(x-mean)**2 for x in l]) / len(l)
        std = variance ** 0.5 #squareroot
        return [(x - mean ) / std for x in l] , std, mean

    def _transform(self):
        #train
        self.x_train_stand, self.x_std, self.x_mean = self._standardize(self.x_train)
        self.y_train_stand, self.y_std, self.y_mean = self._standardize(self.y_train)
        #test
        self.x_test_stand = [(x - self.x_mean)/self.x_std for x in self.x_test]
        self.y_test_stand = [(x - self.y_mean)/self.y_std for x in self.y_test]

    def predict(self,x):
        # f(x) = wx + b
        return self.w * x + self.b

    #method for predicting with real numbers
    def fit(self,x):
        x_train_stand = (x - self.x_mean) / self.x_std
        y_train_stand = self.predict(x_train_stand)
        return y_train_stand * self.y_std + self.y_mean

    def _cost(self,x,y):
        m = len(x)
        return 1/(2*m) * sum([(self.predict(x) - y)**2 for x,y in zip(x,y)])

    def train_cost(self):
        return self._cost(self.x_train_stand,self.y_train_stand)

    def test_cost(self):
        return self._cost(self.x_test_stand,self.y_test_stand)


    def grad(self):
        # C(w,b) = 1/2m * sum((y_hat-y)**2) = 1/2m * sum(((wx+b)-y)**2)
        # dC/dw = (wx+b-y)**2 = 2(wx+b-y) * x -> 1/2m * sum(2x(wx+b-y)) = 1/m * sum(x(wx+b-y)) chain rule derv
        # dC/db = (wx+b-y)**2 = 2(wx+b-y) * 1 -> 1/2m * sum(2(wx+b-y)) = 1/m * sum(wx+b-y)
        m = len(self.x_train_stand)
        self.w_grad = sum(x * (self.predict(x)-y) for x,y in zip(self.x_train_stand,self.y_train_stand)) * 1/m
        self.b_grad = sum(self.predict(x) - y for x,y in zip(self.x_train_stand,self.y_train_stand)) * 1/m


    def train(self,iter=100):
        self._train_test()
        self._transform()
        for i in range(iter):
            self.grad()
            self.w = self.w - self.lr * self.w_grad
            self.b = self.b - self.lr * self.b_grad

    def ols(self):
        self._train_test()
        self._transform()
        #ols formula on real space
        # ols_w = sum((xi-x_mean) * (yi-y_mean)) / sum(xi-x_mean)**2)
        # ols_b = y_mean - w*x_mean
        # ols formula on standardized space (mean is 0)
        # ols_w = sum(x*y) / sum(x**2)
        # ols_b = 0
        self.w = sum([x*y for x,y in zip(self.x_train_stand,self.y_train_stand)]) / sum([x**2 for x in self.x_train_stand])
        self.b = 0


    def test(self):
        print(f"Train Cost: {self.train_cost()}")
        print(f"Test Cost: {self.test_cost()}")
        print("------ Data - Prediction - Actual data --------")
        for i in range(len(self.y_test)):
            print(f"{self.x_test[i]} -- {self.fit(self.x_test[i])} -- {self.y_test[i]}")
