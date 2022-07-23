import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

def call_data_MNIST(batch_size_trainset, batch_size_testset):

    mnist_train = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                              train=True,  # True를 지정하면 훈련 데이터로 다운로드
                              transform=transforms.ToTensor(),  # 텐서로 변환
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                             train=False,  # False를 지정하면 테스트 데이터로 다운로드
                             transform=transforms.ToTensor(),  # 텐서로 변환
                             download=True)

    train_data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size_trainset,
                                              shuffle=True,
                                              drop_last=True)

    test_data_loader =  torch.utils.data.DataLoader(dataset=mnist_test,
                                              batch_size=batch_size_testset,
                                              shuffle=True,
                                              drop_last=True)

    L = [train_data_loader, test_data_loader]

    return L

