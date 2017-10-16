class Model(object):
    def __init__(self,train_op,cost,original,corrupted,deblurred,init):
        self.train_op = train_op
        self.cost = cost
        self.original = original
        self.corrupted = corrupted
        self.deblurred = deblurred
        self.init = init
