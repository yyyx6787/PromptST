import torch
import numpy as np
import time
from engine_crime import trainer, trainerp, trainerf
from Params import args
import random

def seed_torch(seed=523):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def makePrint(name, ep, reses):
    ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
    for metric in reses:
        val = reses[metric]
        ret += '%s = %.4f, ' % (metric, val)
    ret = ret[:-2] + '  '
    return ret

def main():
    # seed_torch(101)
    device = torch.device('cuda:0')
    engine = trainer(device)
    print("start training...", flush=True)
    train_time = []
    bestRes = None
    eval_bestRes = dict()
    eval_bestRes['RMSE'], eval_bestRes['MAE'], eval_bestRes['MAPE'] = 1e6, 1e6, 1e6
    update = False

    for i in range(1, args.epoch+1):
        t1 = time.time()
        metrics, metrics1 = engine.train()
        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
        ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1)
        print(ret)

        test = (i % args.tstEpoch == 0 and i > 0)
        if test:
            res_eval = engine.eval(True, True)
            val_metrics = res_eval['RMSE'] + res_eval['MAE']
            val_best_metrics = eval_bestRes['RMSE'] + eval_bestRes['MAE']
            if (val_metrics) < (val_best_metrics):
                print('%s %.4f, %s %.4f' % ('Val   metrics decrease from', val_best_metrics, 'to', val_metrics))
                eval_bestRes['RMSE'] = res_eval['RMSE']
                eval_bestRes['MAE'] = res_eval['MAE']
                update = True
            reses = engine.eval(False, True)
            print(makePrint('Test', i, reses))
            torch.save(engine.model.state_dict(),
                       args.save + args.data + "/" + "imodel"+ "/" + "_epoch_" + str(i) + "_MAE_" + str(round(reses['MAE'], 2)) + "_MAPE_" + str(
                           round(reses['MAPE'], 2)) + ".pth")
            if update:
                # print(makePrint('Test', i, reses))
                bestRes = reses
                update = False
        print()
        t2 = time.time()
        train_time.append(t2-t1)
    # print("*:", bestRes)
    print(makePrint('Best', args.epoch, bestRes))

def mainp():
    # seed_torch(101)
    device = torch.device('cuda:0')
    engine = trainerp(device)
    print("start training...", flush=True)
    train_time = []
    bestRes = None
    eval_bestRes = dict()
    eval_bestRes['RMSE'], eval_bestRes['MAE'], eval_bestRes['MAPE'] = 1e6, 1e6, 1e6
    update = False

    for i in range(1, args.epoch+1):
        t1 = time.time()
        metrics, metrics1 = engine.train()
        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
        ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1)
        print(ret)

        test = (i % args.tstEpoch == 0 and i > 0)
        if test:
            res_eval = engine.eval(True, True)
            val_metrics = res_eval['RMSE'] + res_eval['MAE']
            # print("&&:", val_metrics)
            val_best_metrics = eval_bestRes['RMSE'] + eval_bestRes['MAE']
            if (val_metrics) < (val_best_metrics):
                print('%s %.4f, %s %.4f' % ('Val   metrics decrease from', val_best_metrics, 'to', val_metrics))
                eval_bestRes['RMSE'] = res_eval['RMSE']
                eval_bestRes['MAE'] = res_eval['MAE']
                update = True
            reses = engine.eval(False, True)
            print(makePrint('Test', i, reses))
            torch.save(engine.pmodel.state_dict(),
                       args.save + args.data + "/"+ "pmodel"+ "/" + "_epoch_" + str(i) + "_MAE_" + str(round(reses['MAE'], 2)) + "_MAPE_" + str(
                           round(reses['MAPE'], 2)) + ".pth")
            if update:
                # print(makePrint('Test', i, reses))
                bestRes = reses
                update = False
        print()
        t2 = time.time()
        train_time.append(t2-t1)
    # print("*:", bestRes)
    print(makePrint('Best', args.epoch, bestRes))
def mainf():
    # seed_torch(101)
    device = torch.device('cuda:0')
    engine = trainerf(device)
    print("start training...", flush=True)
    train_time = []
    bestRes = None
    eval_bestRes = dict()
    eval_bestRes['RMSE'], eval_bestRes['MAE'], eval_bestRes['MAPE'] = 1e6, 1e6, 1e6
    update = False

    for i in range(1, args.epoch+1):
        t1 = time.time()
        metrics, metrics1 = engine.train()
        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
        ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1)
        print(ret)

        test = (i % args.tstEpoch == 0 and i > 0)
        if test:
            res_eval = engine.eval(True, True)
            val_metrics = res_eval['RMSE'] + res_eval['MAE']
            # print("&&:", val_metrics)
            val_best_metrics = eval_bestRes['RMSE'] + eval_bestRes['MAE']
            if (val_metrics) < (val_best_metrics):
                print('%s %.4f, %s %.4f' % ('Val   metrics decrease from', val_best_metrics, 'to', val_metrics))
                eval_bestRes['RMSE'] = res_eval['RMSE']
                eval_bestRes['MAE'] = res_eval['MAE']
                update = True
            reses = engine.eval(False, True)
            print(makePrint('Test', i, reses))
            torch.save(engine.model.state_dict(),
                       args.save + args.data + "/"+ "fmodel"+ "/" + "_epoch_" + str(i) + "_MAE_" + str(round(reses['MAE'], 2)) + "_MAPE_" + str(
                           round(reses['MAPE'], 2)) + ".pth")
            if update:
                # print(makePrint('Test', i, reses))
                bestRes = reses
                update = False
        print()
        t2 = time.time()
        train_time.append(t2-t1)
    # print("*:", bestRes)
    print(makePrint('Best', args.epoch, bestRes))

if __name__ == "__main__":
    t1 = time.time()
    # main() # predefined train
    mainp() # prompt tuning
    # mainf() # fine tuning
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))