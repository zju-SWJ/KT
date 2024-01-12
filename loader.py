import os
import torch.utils.data as data
import torch

class TrainDataset(data.Dataset):
    def __init__(self, base_path='/data/train', num=None):
        super().__init__()
        self.input_files = []
        self.target_files = []

        for i in range(num):
            self.input_files.append(os.path.join(base_path, 'inputs', str(i) + '.pth'))
            self.target_files.append(os.path.join(base_path, 'labels', str(i) + '.pth'))

        print('load', len(self.input_files), 'training data')

    def __getitem__(self, index):
        input_file, target_file = self.input_files[index], self.target_files[index]

        input_ckpt = torch.load(input_file, 'cpu')
        target_ckpt = torch.load(target_file, 'cpu')

        input = [input_ckpt['conv1.weight'].data, input_ckpt['conv2.weight'].data, 
                    input_ckpt['bn1.weight'].data, input_ckpt['bn1.bias'].data, input_ckpt['bn1.running_mean'].data, input_ckpt['bn1.running_var'].data,
                    input_ckpt['bn2.weight'].data, input_ckpt['bn2.bias'].data, input_ckpt['bn2.running_mean'].data, input_ckpt['bn2.running_var'].data]
        target = [target_ckpt['conv1.weight'].data, target_ckpt['conv2.weight'].data]

        return input, target

    def __len__(self):
        return len(self.input_files)

class ValDataset(data.Dataset):
    def __init__(self, base_path='/data/val', num=None):
        super().__init__()
        self.begin_files = []
        self.complex_files = []
        self.last_files = []

        for i in range(num):
            self.begin_files.append(os.path.join(base_path, 'begin', str(i) + '.pth'))
            self.complex_files.append(os.path.join(base_path, 'complex', str(i) + '.pth'))
            self.last_files.append(os.path.join(base_path, 'last', str(i) + '.pth'))

        print('load', len(self.begin_files), 'validation data')

    def __getitem__(self, index):
        begin_file, complex_file, last_file = self.begin_files[index], self.complex_files[index], self.last_files[index]

        ckpt = torch.load(complex_file, 'cpu')
        
        input = [ckpt['conv1.weight'].data, ckpt['conv2.weight'].data,
                ckpt['bn1.weight'].data, ckpt['bn1.bias'].data, ckpt['bn1.running_mean'].data, ckpt['bn1.running_var'].data,
                ckpt['bn2.weight'].data, ckpt['bn2.bias'].data, ckpt['bn2.running_mean'].data, ckpt['bn2.running_var'].data]
        
        return begin_file, input, last_file

    def __len__(self):
        return len(self.begin_files)