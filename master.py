import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import model_CNN as CNN
import model_DNN as DNN
from run import*
from data_loader_MNIST import*
# gpu or cpu 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print(torch.cuda.is_available())

# train을 위한 초기변수
learning_rate = 0.001
epochs = 5
batch_size = 100

# CNN 모델 정의
model = CNN.CNN().to(device)

# loss function 및 optimizer 정의
loss = torch.nn.CrossEntropyLoss().to(device)    # CEL 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data = call_data_MNIST(100,20)

run(model,loss,optimizer,device,train_loader = data[0], test_loader = data[1],epochs = epochs,log_interval=100)