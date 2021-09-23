import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import MyDataset


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据转换
#image转换
x_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask转换
y_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor()
])
#此处修改训练的epoch
def train_model(model, criterion, optimizer, dataload, num_epochs=99):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型
def train(batch_size):
    model = Unet(3, 1).to(device)
    batch_size = batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = MyDataset("u_net_master/altrasound/train",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(ckpt):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(ckpt,map_location='cpu'))
    liver_dataset = MyDataset("unet_master/altrasound/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()

    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x).sigmoid()
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(1)
        plt.show()


if __name__ == '__main__':
#设置训练时的batch_size
    batch_size = 4
    train(batch_size)

#训练后进行测试
    #ckpt = "/home/zzzxxx/Desktop/u_net_liver-master/weights_8.pth"
    #test(ckpt)



