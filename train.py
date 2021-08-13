import wandb
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def train(model, train_loader, eval_loader, n_epoches, optimizer, threshold=0.7, eval_per_step=10, use_wandb=False, use_distr=True, device="cuda:0", acc_step=1):
    if use_wandb:
        wandb.watch(model, log_freq=eval_per_step)
    if use_distr:
        scaler = GradScaler()
    for epoch in range(n_epoches):
        print("epoch " + str(epoch+1))
        cnt = 0
        tot_loss = 0
        
        model.train()
        print("train")
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
                    acc = evaluate(model, eval_loader, threshold)
                    ac2 = evaluate(model, train_loader,threshold)
                    print("loss"+str(tot_loss/eval_per_step))
                    if use_wandb :
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
            '''
            with torch.no_grad():
                p, n = model.get_avg(model.forward_once(inputs), labels)
            #print(p, n)
            '''
            #optimizer.step()
        #save(model, epoch)


def evaluate(model, loader, threshold):
    return 0
