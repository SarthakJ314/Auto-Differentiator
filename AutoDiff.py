import numpy as np

class Number:
    def __init__(self, val, op = None, p1 = None, p2 = None):
        self.x = val
        self.op = op
        self.grad = 0.0
        self.deg = 0
        self.const = False
        self.p = []
        if p1:
            p1.deg += 1
            self.p = [p1]
        if p2:
            p2.deg += 1
            self.p.append(p2)

    def toString(self):
        return str(self.x) + " " + str(self.grad)

    def __add__(self, x):
        self, x = check(self, x)
        return Number(x.x + self.x, "add", self, x)

    def __radd__(self, x):
        self, x = check(self, x)
        return Number(x.x + self.x, "add", x, self)

    def __mul__(self, x):
        self, x = check(self, x)
        return Number(x.x * self.x, "mul", self, x)

    def __rmul__(self, x):
        self, x = check(self, x)
        return Number(x.x * self.x, "mul", x, self)

    def __sub__(self, x):
        self, x = check(self, x)
        return Number(self.x - x.x, "sub", self, x)

    def __rsub__(self, x):
        self, x = check(self, x)
        return Number(x.x - self.x, "sub", x, self)

    def __truediv__(self, x):
        self, x = check(self, x)
        return Number(self.x / x.x, "truediv", self, x)

    def __rtruediv__(self, x):
        self, x = check(self, x)
        return Number(x.x / self.x, "truediv", x, self)

    def __pow__(self, x):
        self, x = check(self, x)
        return Number(self.x**x.x, "pow", self, x)

    def __rpow__(self, x):
        self, x = check(self, x)
        return Number(x.x**self.x, "pow", x, self)

def sin(x):
    return Number(np.sin(x.x), "sin", x)

def cos(x):
    return Number(np.cos(x.x), "cos", x)

def tan(x):
    return sin(x) / cos(x)

def csc(x):
    return 1 / sin(x)

def sec(x):
    return 1 / cos(x)

def cot(x):
    return 1 / tan(x)

def log(x):
    return Number(np.log(x.x), "ln", x)

def sigmoid(x):
    return 1 / (1 + np.e**(-1 * x))

def check(x, y):
    xx, yy = x, y
    if not isinstance(x, Number):
        xx = Number(x)
        xx.const = True
    if not isinstance(y, Number):
        yy = Number(y)
        yy.const = True
    return xx, yy

back_func1 = {
    "add" : lambda x, y, d: d,
    "sub" : lambda x, y, d: d,
    "mul" : lambda x, y, d: d * y.x,
    "truediv" : lambda x, y, d: d / y.x,
    "pow" : lambda x, y, d: d * y.x * x.x**(y.x-1),
    "sin" : lambda x, d: d * np.cos(x.x),
    "cos" : lambda x, d: d * -1 * np.sin(x.x),
    "ln" : lambda x, d: d / x.x
}

back_func2 = {
    "add" : lambda x, y, d: d,
    "sub" : lambda x, y, d: -1 * d,
    "mul" : lambda x, y, d: d * x.x,
    "truediv" : lambda x, y, d: -1 * d * x.x / y.x**2,
    "pow" : lambda x, y, d: d * x.x**(y.x) * np.log(x.x)
}

def BackClear(x):
    q = [x]
    while len(q) > 0:
        cur = q.pop(0)
        for i in cur.p:
            q.append(i)
        if len(cur.p) and cur.const:
            del cur

def BackProp(x):
    queue = [x]
    toplOrder = []
    degs = {}
    while len(queue) > 0:
        curr = queue.pop(0)
        toplOrder.append(curr)
        curr.grad = 0
        for i in curr.p:
            if i not in degs:
                degs[i] = i.deg
            degs[i] -= 1
            if degs[i] == 0:
                queue.append(i)
    x.grad = 1
    for cur in toplOrder:
        if len(cur.p) == 0 or cur.op == None:
            continue
        if len(cur.p) == 1 and cur.p[0].const == False:
            cur.p[0].grad += back_func1[cur.op](cur.p[0], cur.grad)        
        if len(cur.p) == 2 and cur.p[0].const == False:
            cur.p[0].grad += back_func1[cur.op](cur.p[0], cur.p[1], cur.grad)
        if len(cur.p) == 2 and cur.p[1].const == False:
            cur.p[1].grad += back_func2[cur.op](cur.p[0], cur.p[1], cur.grad)
