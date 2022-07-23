from train import*
from test import*

def run(model,loss,optimizer,device,train_loader,test_loader,epochs,log_interval=100):
    for epoch in range(epochs):
        train(model, loss, optimizer, device, train_loader, log_interval)
        test(model, device, test_loader)


