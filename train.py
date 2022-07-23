import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

def train(model,loss_fn,optimizer,device,train_data_loader,log_interval=100):
    model.train()

    for batch, (data, target) in enumerate(train_data_loader):  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded

        # data = image, target = label
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad() # grad 0으로 초기화
        output = model(data) # foward propagation
        loss = loss_fn(output, target) # loss
        loss.backward() # back propagation
        optimizer.step() # weight update

        if batch % log_interval == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 batch * len(data), len(train_data_loader.dataset),
                100. * batch / len(train_data_loader), loss.item()))


