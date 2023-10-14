import numpy as np

class Confusion():
    def __init__(self, mat):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError('Confusion matrix should be a squre matrix.')
        self.mat = mat
        
    def TP(self, idx=-1):
        return self.mat[idx,idx] if idx >= 0 else np.diag(self.mat)
    
    def FP(self, idx=-1):
        return self.mat[:,idx].sum() - self.mat[idx,idx] if idx >= 0 else np.sum(self.mat,0) - np.diag(self.mat)
    
    def FN(self, idx=-1):
        return self.mat[idx,:].sum() - self.mat[idx,idx] if idx >= 0 else np.sum(self.mat,1) - np.diag(self.mat)
    
    def TN(self, idx=-1):
        return self.mat.sum() - self.TP(idx) - self.FP(idx) - self.FN(idx)
        
    def recall(self, idx=-1):
        return self.TP(idx) / (self.TP(idx) + self.FN(idx))
    
    def precision(self, idx=-1):
        return self.TP(idx) / (self.TP(idx) + self.FP(idx))
    
    def f1(self, idx=-1):
        return 2 * self.TP(idx) / (2 * self.TP(idx) + self.FP(idx) + self.FN(idx))
    
    def accuracy(self):
        return np.diag(self.mat).sum() / self.mat.sum()
    
    def everything(self):
        ml = ['recall', 'precision', 'f1', 'accuracy']
        return dict([(func, getattr(self, func)().mean()) for func in ml])

    def everything_str(self):
        ret = ' | '.join(['%s:%.4f'%(k,v) for k, v in self.everything().items()])
        return ret