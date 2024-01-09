import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import net
import os
from absl import app, flags
import copy

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_string('gpu_id', '0', help='gpu id')
flags.DEFINE_integer('num_epochs', 10, help='number of epochs')
flags.DEFINE_float('lr', 0.01, help='learning rate')
flags.DEFINE_integer('num_data', 1, help='number of data')
flags.DEFINE_integer('start_index', 0, help='start index')
flags.DEFINE_string('data_root', '/data/nobn', help='data root')
device = torch.device('cuda:0')


def train_and_eval(index):
    print('Data index:', index)
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(), transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    assert len(trainset) == 60000 and len(testset) == 10000

    BeginBlock = net.BeginBlock()
    ComplexBlock = net.BasicBlock()
    LastBlock = net.LastBlock()

    ComplexNet = nn.Sequential(BeginBlock, ComplexBlock, LastBlock).to(device)
    ComplexOptimizer = optim.SGD(ComplexNet.parameters(), lr=FLAGS.lr, momentum=0.9)

    for epoch in range(FLAGS.num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = ComplexNet(inputs)
            loss = F.cross_entropy(outputs, labels)
            ComplexOptimizer.zero_grad()
            loss.backward()
            ComplexOptimizer.step()
        with torch.no_grad():
            ComplexNet.eval()
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                out = ComplexNet(images)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            ComplexNet.train()
            print('Epoch {}, Complex Test Accuracy {}%'.format(epoch, 100 * correct / total))
    print('--------------------------------------------------')
    ComplexNet.eval()
    copy_begin_block_dict = copy.deepcopy(BeginBlock.state_dict())
    copy_last_block_dict = copy.deepcopy(LastBlock.state_dict())

    SimpleBlock = net.SimpleBlock_NoBn()
    SimpleNet = nn.Sequential(BeginBlock, SimpleBlock, LastBlock).to(device)
    SimpleOptimizer = optim.SGD(SimpleBlock.parameters(), lr=FLAGS.lr, momentum=0.9)

    with torch.no_grad():
        SimpleBlock.eval()
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            out = SimpleNet(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        SimpleBlock.train()
        print('Before Train, Test Accuracy {}%'.format(100 * correct / total))

    for epoch in range(FLAGS.num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = SimpleNet(inputs)
            loss = F.cross_entropy(outputs, labels)
            SimpleOptimizer.zero_grad()
            loss.backward()
            SimpleOptimizer.step()
        with torch.no_grad():
            SimpleBlock.eval()
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                out = SimpleNet(images)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            SimpleBlock.train()
            print('Epoch {}, Simple Test Accuracy {}%'.format(epoch, 100 * correct / total))
    for key in BeginBlock.state_dict().keys():
        assert torch.equal(BeginBlock.state_dict()[key], copy_begin_block_dict[key])
    for key in LastBlock.state_dict().keys():
        assert torch.equal(LastBlock.state_dict()[key], copy_last_block_dict[key])
    if not os.path.exists(os.path.join(FLAGS.data_root, 'inputs')):
        os.makedirs(os.path.join(FLAGS.data_root, 'inputs'))
    if not os.path.exists(os.path.join(FLAGS.data_root, 'labels')):
        os.makedirs(os.path.join(FLAGS.data_root, 'labels'))
    if not os.path.exists(os.path.join(FLAGS.data_root, 'backup_begin')):
        os.makedirs(os.path.join(FLAGS.data_root, 'backup_begin'))
    if not os.path.exists(os.path.join(FLAGS.data_root, 'backup_last')):
        os.makedirs(os.path.join(FLAGS.data_root, 'backup_last'))
    torch.save(ComplexBlock.state_dict(), FLAGS.data_root + '/inputs/' + str(index) + '.pth')
    torch.save(SimpleBlock.state_dict(), FLAGS.data_root + '/labels/' + str(index) + '.pth')
    torch.save(BeginBlock.state_dict(), FLAGS.data_root + '/backup_begin/' + str(index) + '.pth')
    torch.save(LastBlock.state_dict(), FLAGS.data_root + '/backup_last/' + str(index) + '.pth')

def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    torch.backends.cudnn.benchmark = True
    for i in range(FLAGS.start_index, FLAGS.start_index + FLAGS.num_data):
        if os.path.exists(FLAGS.data_root + '/inputs/' + str(i) + '.pth') and os.path.exists(FLAGS.data_root + '/labels/' + str(i) + '.pth') and \
        os.path.exists(FLAGS.data_root + '/backup_begin/' + str(i) + '.pth') and os.path.exists(FLAGS.data_root + '/backup_last/' + str(i) + '.pth'):
            continue
        train_and_eval(i)

if __name__ == '__main__':
    app.run(main)