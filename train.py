from torch.distributed.distributed_c10d import get_rank
from torch.utils.data import distributed
import wandb
import pickle
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def train(model, train_loader, eval_loader, n_epoches, optimizer, threshold=0.7, eval_per_step=10, use_wandb=False, use_distr=True, device="cuda:0", acc_step=1):
    if use_wandb:
        if use_distr:
            if torch.distributed.get_rank()==0:
                wandb.watch(model, log_freq=eval_per_step)
        else:
            wandb.watch(model, log_freq=eval_per_step)
    if use_distr:
        scaler = GradScaler()
    for epoch in range(n_epoches):
        if (use_distr and torch.distributed.get_rank()==0) or (not use_distr) :
            print("epoch " + str(epoch+1))
            print("train")
        cnt = 0
        tot_loss = 0
        
        model.train()
        
        #pbar = tqdm(train_loader, unit="batch")
        #for toks1, toks2 in pbar:
        for i, (toks1, toks2) in enumerate(train_loader):
            if use_distr:
                #if device!=torch.device("cuda",0):
                toks1, toks2 = toks1.to(device), toks2.to(device)
            #else:
            #    toks1, toks2 = toks1[:4].to(device), toks2[:4].to(device)
            else:
                toks1, toks2 = toks1.cuda(non_blocking=True), toks2.cuda(non_blocking=True)
            
            #print(toks1[:5])
            #print("data shape ", toks1.shape, toks2.shape)
            cnt += 1
            
            if cnt%eval_per_step == 0 :
                if cnt%(eval_per_step*1)==0:
                    acc = evaluate(model, eval_loader, threshold, use_distr=use_distr)
                    ac2 = evaluate(model, train_loader,threshold, use_distr=use_distr)
                    
                    acc = acc.view(-1).cpu().item()
                    if (use_distr and torch.distributed.get_rank()==0) or not use_distr:
                        print("acc: ", acc)
                        print("loss"+str(tot_loss/eval_per_step))
                    if use_wandb :
                        if use_distr:
                            if torch.distributed.get_rank()==0:
                                wandb.log({"train/train-acc": ac2, "train/eval-acc": acc,"train/loss": tot_loss/eval_per_step})
                        else :
                            wandb.log({"train/train-acc": ac2, "train/eval-acc": acc,"train/loss": tot_loss/eval_per_step}) 
                    tot_loss = 0
                model.train()
            
            
            if cnt%acc_step==0:
                if use_distr:
                    with autocast():
                        out = model((toks1, toks2))
                    loss = model.module.get_loss(out)
                    tot_loss += loss.detach().cpu().item()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    optimizer.zero_grad()
                    loss = model.get_loss(model((toks1, toks2)))
                    tot_loss += loss.detach().cpu().item()
                    loss.backward()
                    optimizer.step()
            else:
                if use_distr:
                    with model.no_sync():
                        with autocast():
                            out = model((toks1, toks2))
                        loss = model.module.get_loss(out)
                        scaler.scale(loss).backward()
                else:
                    loss = model.get_loss(model((toks1, toks2)))
                    loss.backward()
        save(model, epoch)


def evaluate(model, loader, threshold=0.7, use_distr=False):
    model.eval()
    correct = torch.tensor([0]).cuda()
    total = torch.tensor([0]).cuda()
    for i, (toks1, toks2) in enumerate(loader):
        if i>40:
            break
        if use_distr:
            with torch.no_grad():
                out = model((toks1, toks2))
            right, num = model.module.get_acc(out)
            correct += right
            total += num
        else:
            right, num = model.get_acc(model((toks1, toks2)))
            correct += right
            total += num
    torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    return correct / total

def do_embedding(model, loader, path, use_distr=False, device="cuda:0"):
    model.eval()
    res = []
    for i, (ids, toks) in enumerate(loader):
        toks = toks.to(device)
        with torch.no_grad():
            if use_distr:
                out = model.module.forward_once(toks)
            else:
                out = model.forward_once(toks)
        out = out.cpu().numpy()
        res.extend(
            [(ids[i], out[i]) for i in range(out.shape[0])]
        )
    with open(path[:-4], mode="wb") as f:
        pickle.dump(res, f)

def save(model, epoch):
    torch.save(model.state_dict(), './continue_train/'+str(epoch)+'.pth')
