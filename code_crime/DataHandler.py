import pickle
import numpy as np
import torch
from util_MTGNN import asym_adj
from Params import args

class DataHandler:
    def __init__(self):
        if args.data == 'NYC':
            predir = 'Datasets/NYC_crime/'
        elif args.data == 'CHI':
            predir = 'Datasets/CHI_crime/'
        else:
            predir = None
        self.predir = predir
        with open(predir + 'trn.pkl', 'rb') as fs:
            trnT = pickle.load(fs)
        with open(predir + 'val.pkl', 'rb') as fs:
            valT = pickle.load(fs)
        with open(predir + 'tst.pkl', 'rb') as fs:
            tstT = pickle.load(fs)
        print("self.predir", self.predir)
        args.row, args.col, _, args.offNum = trnT.shape
        args.areaNum = args.row * args.col
        args.trnDays = trnT.shape[2]
        args.valDays = valT.shape[2]
        args.tstDays = tstT.shape[2]
        args.decay_step = args.trnDays//args.batch

        self.trnT1 = np.reshape(trnT, [args.areaNum, -1, args.offNum])
        self.valT = np.reshape(valT, [args.areaNum, -1, args.offNum])
        self.tstT = np.reshape(tstT, [args.areaNum, -1, args.offNum])
        # print("self.trnT1:", self.trnT1.shape)
        # println()
        self.trnT = self.trnT1[:,:519, :]  #518-->NW, 609-90-->CHI
        self.prmT = self.trnT1[:,519:, :]
        self.mean = np.mean(trnT)
        self.std = np.std(trnT)
        self.mask1, self.mask2, self.mask3, self.mask4 = self.getSparsity()
        self.getTestAreas()
        print('Row:', args.row, ', Col:', args.col)
        print('Sparsity:', np.sum(trnT!=0) / np.reshape(trnT, [-1]).shape[0])

    def zScore(self, data):
        return (data - self.mean) / self.std

    def zInverse(self, data):
        return data * self.std + self.mean

    def getSparsity(self):
        data = self.tstT
        print(data.shape)
        day = data.shape[1]
        mask = 1 * (data > 0)
        p1 = np.zeros([data.shape[0], data.shape[2]])
        for cate in range(4):
            for region in range(mask.shape[0]):
                p1[region, cate] = np.sum(mask[region, :, cate], axis=0) / day
        mask1 = np.zeros_like(p1)
        mask2 = np.zeros_like(p1)
        mask3 = np.zeros_like(p1)
        mask4 = np.zeros_like(p1)
        for cate1 in range(4):
            for region1 in range(mask.shape[0]):
                if p1[region1, cate1] > 0 and p1[region1, cate1] <= 0.25:
                    mask1[region1, cate1] = 1
                elif p1[region1, cate1] > 0.25 and p1[region1, cate1] <= 0.5:
                    mask2[region1, cate1] = 1
                elif p1[region1, cate1] > 0.5 and p1[region1, cate1] <= 0.75:
                    mask3[region1, cate1] = 1
                elif p1[region1, cate1] > 0.75 and p1[region1, cate1] <= 1:
                    mask4[region1, cate1] = 1
        return mask1, mask2, mask3, mask4

    @classmethod
    def idEncode(cls, x, y):
        return x * args.col + y

    @classmethod
    def idDecode(cls, node):
        return node // args.col, node % args.col

    def constructGraph(self):
        mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
        my = [0, -1, 0, 1, -1, 1, -1, 1, 0]

        def illegal(x, y):
            return x < 0 or y < 0 or x >= args.row or y >= args.col

        W = np.zeros((args.areaNum, args.areaNum))
        for i in range(args.row):
            for j in range(args.col):
                n1 = self.idEncode(i, j)
                for k in range(len(mx)):
                    temx = i + mx[k]
                    temy = j + my[k]
                    if illegal(temx, temy):
                        continue
                    n2 = self.idEncode(temx, temy)
                    W[n1, n2] = 1
        return W


    def Wavenet_Graph(self, adjdata):
        adj = [asym_adj(adjdata), asym_adj(np.transpose(adjdata))]
        supports = [torch.tensor(i).to(args.device) for i in adj]

        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]

        if args.aptonly:
            supports = None

        return supports, adjinit

    def getTestAreas(self):
        posTimes = np.sum(1 * (self.trnT!=0), axis=1)
        percent = posTimes / args.trnDays
        self.tstLocs = (percent > -1) * 1