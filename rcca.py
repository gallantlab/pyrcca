import numpy as np
from scipy.linalg import eigh
import h5py

class _CCABase(object):
    def __init__(self, numCV = None, reg = None, regs = None, numCC = None, numCCs = None, kernelcca = True, ktype = None, verbose = False, select = 0.2, cutoff = 1e-15, gausigma = 1.0, degree = 2):
        self.numCV = numCV
        self.reg = reg
        self.regs = regs
        self.numCC = numCC
        self.numCCs = numCCs
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.select = select
        self.gausigma = gausigma
        self.degree = degree
        if self.kernelcca and self.ktype == None:
            self.ktype = "linear"
        self.verbose = verbose

    def train(self, data):
        nT = data[0].shape[0]
        if self.verbose:
            if self.kernelcca:
                print("Training CCA, %s kernel, regularization = %0.4f, %d components" % (self.ktype, self.reg, self.numCC))
            else:
                print("Training CCA, regularization = %0.4f, %d components" % (self.reg, self.numCC))
        comps = kcca(data, self.reg, self.numCC, kernelcca = self.kernelcca, ktype = self.ktype, gausigma = self.gausigma, degree = self.degree)
        self.cancorrs, self.ws, self.comps = recon(data, comps, kernelcca = self.kernelcca)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self

    def validate(self, vdata):
        vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        if not hasattr(self, 'ws'):
            raise NameError("Algorithm needs to be trained!")
        self.preds, self.corrs = predict(vdata, self.ws, self.cutoff)
        return self.corrs

    def compute_ev(self, vdata):
        nD = len(vdata)
        nT = vdata[0].shape[0]
        nC = self.ws[0].shape[1]
        nF = [d.shape[1] for d in vdata]
        self.ev = [np.zeros((nC, f)) for f in nF]
        for cc in range(nC):
            ccs = cc+1
            if self.verbose:
                print("Computing explained variance for component #%d" % ccs)
            preds, corrs = predict(vdata, [w[:, ccs-1:ccs] for w in self.ws], self.cutoff)
            resids = [abs(d[0]-d[1]) for d in zip(vdata, preds)]
            for s in range(nD):
                ev = abs(vdata[s].var(0) - resids[s].var(0))/vdata[s].var(0)
                ev[np.isnan(ev)] = 0.
                self.ev[s][cc] = ev
        return self.ev

    def save(self, filename):
        h5 = h5py.File(filename, "a")
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, list):
                    for di in range(len(value)):
                        grpname = "dataset%d" % di
                        dgrp = h5.require_group(grpname)
                        try:
                            dgrp.create_dataset(key, data=value[di])
                        except RuntimeError:
                            del h5[grpname][key]
                            dgrp.create_dataset(key, data=value[di])
                else:
                    h5.attrs[key] = value
        h5.close()

    def load(self, filename):
        h5 = h5py.File(filename, "a")
        for key, value in h5.attrs.items():
            setattr(self, key, value)
        for di in range(len(h5.keys())):
            ds = "dataset%d" % di
            for key, value in h5[ds].items():
                if di == 0:
                    setattr(self, key, [])
                self.__getattribute__(key).append(value.value)

class CCACrossValidate(_CCABase):
    '''Attributes:
        numCV - number of crossvalidation folds
        reg - array of regularization parameters. Default is np.logspace(-3, 1, 10)
        numCC - list of numbers of canonical dimensions to keep. Default is np.range(5, 10).
        kernelcca - True if using a kernel (default), False if not kernelized.
        ktype - type of kernel if kernelcca == True (linear or gaussian). Default is linear.
        verbose - True is default

    Results:
        ws - canonical weights
        comps - canonical components
        cancorrs - correlations of the canonical components on the training dataset
        corrs - correlations on the validation dataset
        preds - predictions on the validation dataset
        ev - explained variance for each canonical dimension
    '''
    def __init__(self, numCV = None, regs = None, numCCs = None, kernelcca = True, ktype = None, verbose = True, select = 0.2, cutoff = 1e-15, gausigma = 1.0, degree = 2):
        numCV = 10 if numCV is None else numCV
        regs = np.array(np.logspace(-3, 1, 10)) if regs is None else regs
        numCCs = np.arange(5, 10) if numCCs is None else numCCs
        super(CCACrossValidate, self).__init__(numCV = numCV, regs = regs, numCCs = numCCs, kernelcca = kernelcca, ktype = ktype, verbose = verbose, select = select, cutoff = cutoff, gausigma = gausigma, degree = degree)

    def train(self, data):
        """
        Train CCA for a set of regularization coefficients and/or numbers of CCs
        data - list of training data matrices (number of samples X number of features). Number of samples has to match across datasets.
        """
        nT = data[0].shape[0]
        chunklen = 10 if nT > 50 else 1
        nchunks = int(0.2*nT/chunklen)
        allinds = range(nT)
        indchunks = zip(*[iter(allinds)]*chunklen)
        corr_mat = np.zeros((len(self.regs), len(self.numCCs)))
        selection = int(self.select*min([d.shape[1] for d in data]))
        if selection == 0:
            selection = 1
        for ri, reg in enumerate(self.regs):
            for ci, numCC in enumerate(self.numCCs):
                corr_mean = 0
                for cvfold in range(self.numCV):
                    if self.verbose:
                        if self.kernelcca:
                            print("Training CV CCA, %s kernel, regularization = %0.4f, %d components, fold #%d" % (self.ktype, reg, numCC, cvfold+1))
                        else:
                            print("Training CV CCA, regularization = %0.4f, %d components, fold #%d" % (reg, numCC, cvfold+1))
                    np.random.shuffle(indchunks)
                    heldinds = [ind for chunk in indchunks[:nchunks] for ind in chunk]
                    notheldinds = list(set(allinds)-set(heldinds))
                    comps = kcca([d[notheldinds] for d in data], reg, numCC, kernelcca = self.kernelcca, ktype=self.ktype, gausigma = self.gausigma, degree = self.degree)
                    cancorrs, ws, ccomps = recon([d[notheldinds] for d in data], comps, kernelcca = self.kernelcca)
                    preds, corrs = predict([d[heldinds] for d in data], ws, self.cutoff)
                    corrs_idx = [np.argsort(cs)[::-1] for cs in corrs]
                    corr_mean += np.mean([corrs[corri][corrs_idx[corri][:selection]].mean() for corri in range(len(corrs))])
                corr_mat[ri, ci] = corr_mean/self.numCV
        best_ri, best_ci = np.where(corr_mat == corr_mat.max())
        self.best_reg = self.regs[best_ri[0]]
        self.best_numCC = self.numCCs[best_ci[0]]
        comps = kcca(data, self.best_reg, self.best_numCC, kernelcca = self.kernelcca, ktype = self.ktype, gausigma = self.gausigma, degree = self.degree)
        self.cancorrs, self.ws, self.comps = recon(data, comps, kernelcca = self.kernelcca)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self

class CCA(_CCABase):
    '''Attributes:
        reg - regularization parameters. Default is 0.1.
        numCC - number of canonical dimensions to keep. Default is 10.
        kernelcca - True if using a kernel (default), False if not kernelized.
        ktype - type of kernel if kernelcca == True (linear or gaussian). Default is linear.
        verbose - True is default

    Results:
        ws - canonical weights
        comps - canonical components
        cancorrs - correlations of the canonical components on the training dataset
        corrs - correlations on the validation dataset
        preds - predictions on the validation dataset
        ev - explained variance for each canonical dimension
    '''
    def __init__(self, reg = 0., numCC = 10, kernelcca = True, ktype = None, verbose = True, cutoff = 1e-15):
        super(CCA, self).__init__(reg = reg, numCC = numCC, kernelcca = kernelcca, ktype = ktype, verbose = verbose, cutoff = cutoff)

    def train(self, data):
        return super(CCA, self).train(data)

def predict(vdata, ws, cutoff = 1e-15):
    '''Get predictions for each dataset based on the other datasets and weights. Find correlations with actual dataset.'''
    iws = [np.linalg.pinv(w.T, rcond = cutoff) for w in ws]
    ccomp = _listdot([d.T for d in vdata], ws)
    ccomp = np.array(ccomp)
    preds = []
    corrs = []

    for dnum in range(len(vdata)):
        idx = np.ones((len(vdata),))
        idx[dnum] = False
        proj = ccomp[idx>0].mean(0)
        pred = np.dot(iws[dnum], proj.T).T
        pred = np.nan_to_num(_zscore(pred))
        preds.append(pred)
        cs = np.nan_to_num(_rowcorr(vdata[dnum].T, pred.T))
        corrs.append(cs)
    return preds, corrs

def kcca(data, reg = 0., numCC=None, kernelcca = True, ktype = "linear", gausigma = 1.0, degree = 2):
    '''Set up and solve the eigenproblem for the data in kernel and specified reg
    '''
    if kernelcca:
        kernel = [_make_kernel(d, ktype = ktype, gausigma = gausigma, degree = degree) for d in data]
    else:
        kernel = [d.T for d in data]

    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

    # Get the kernel auto- and cross-covariance matrices
    if kernelcca:
        crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]
    else:
        crosscovs = [np.dot(ki, kj.T).T for ki in kernel for kj in kernel]

    # Allocate LH and RH:
    LH = np.zeros((np.sum(nFs), np.sum(nFs)))
    RH = np.zeros((np.sum(nFs), np.sum(nFs)))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(len(kernel)):
        RH[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1])), int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1]))] = crosscovs[i*(len(kernel)+1)] + reg*np.eye(nFs[i])
        for j in range(len(kernel)):
            if i !=j:
                LH[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1])), int(np.sum(nFs[:j])):int(np.sum(nFs[:j+1]))] = crosscovs[len(kernel)*j+i]

    LH = (LH+LH.T)/2.
    RH = (RH+RH.T)/2.

    maxCC = LH.shape[0]

    r, Vs = eigh(LH, RH, eigvals = (maxCC-numCC, maxCC-1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(len(kernel)):
        comp.append(Vs[int(np.sum(nFs[:i])):int(np.sum(nFs[:i+1])), :numCC])
    
    return comp

def recon(data, comp, corronly=False, kernelcca = True):
    nT = data[0].shape[0]
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp

def _zscore(d): return (d-d.mean(0))/d.std(0)
def _demean(d): return d-d.mean(0)
def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]
def _listcorr(a):
    '''Returns pairwise row correlations for all items in array as a list of matrices
    '''
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j>i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0,1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs
def _rowcorr(a, b):
    '''Correlations between corresponding matrix rows
    '''
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0,1]
    return cs

def _make_kernel(d, normalize = True, ktype = "linear", gausigma = 1.0, degree = 2):
    '''Makes a kernel for data d
      If ktype is "linear", the kernel is a linear inner product
      If ktype is "gaussian", the kernel is a Gaussian kernel with sigma = gausigma
      If ktype is "poly", the kernel is a polynomial kernel with degree = degree
    '''
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == "linear":
        kernel = np.dot(cd,cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / 2*gausigma ** 2)
    elif ktype == "poly":
        kernel = np.dot(cd, cd.T)**degree
    kernel = (kernel+kernel.T)/2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel
