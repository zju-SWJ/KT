import os
import torch.utils.data as data
import torch

class TrainDataset(data.Dataset):
    def __init__(self, base_path='/data/train', num=None):
        super().__init__()
        self.inputs = []
        self.targets = []

        inputs_file_path = os.path.join(base_path, 'inputs')
        targets_file_path = os.path.join(base_path, 'labels')
        
        for i in range(num):
            self.inputs.append(os.path.join(inputs_file_path, str(i) + '.pth'))
            self.targets.append(os.path.join(targets_file_path, str(i) + '.pth'))

        print('load', len(self.inputs), 'training data')

    def __getitem__(self, index):
        input, target = self.inputs[index], self.targets[index]
        
        input_name = (input.split('/')[-1]).strip()
        target_name = (target.split('/')[-1]).strip()
        assert input_name == target_name

        T_ckpt = torch.load(input, 'cpu')
        S_ckpt = torch.load(target, 'cpu')

        input = (T_ckpt['conv1.weight'].data, T_ckpt['conv2.weight'].data, 
                    T_ckpt['bn1.weight'].data, T_ckpt['bn1.bias'].data, T_ckpt['bn1.running_mean'].data, T_ckpt['bn1.running_var'].data,
                    T_ckpt['bn2.weight'].data, T_ckpt['bn2.bias'].data, T_ckpt['bn2.running_mean'].data, T_ckpt['bn2.running_var'].data)
        target = (S_ckpt['conv1.weight'].data, S_ckpt['conv2.weight'].data)

        return input, target

    def __len__(self):
        return len(self.inputs)

class ValDataset(data.Dataset):
    def __init__(self, base_path='/data/val', num=None):
        super().__init__()
        self.begin_ckpts = []
        self.complex_ckpts = []
        self.last_ckpts = []

        begin_path = os.path.join(base_path, 'begin')
        complex_path = os.path.join(base_path, 'complex')
        last_path = os.path.join(base_path, 'last')

        for i in range(num):
            self.begin_ckpts.append(os.path.join(begin_path, str(i) + '.pth'))
            self.complex_ckpts.append(os.path.join(complex_path, str(i) + '.pth'))
            self.last_ckpts.append(os.path.join(last_path, str(i) + '.pth'))

        print('load', len(self.begin_ckpts), 'validation data')

    def __getitem__(self, index):
        begin_ckpt, complex_ckpt, last_ckpt = self.begin_ckpts[index], self.complex_ckpts[index], self.last_ckpts[index]

        begin_ckpt_name = (begin_ckpt.split('/')[-1]).strip()
        complex_ckpt_name = (complex_ckpt.split('/')[-1]).strip()
        last_ckpt_name = (last_ckpt.split('/')[-1]).strip()
        assert begin_ckpt_name == complex_ckpt_name
        assert begin_ckpt_name == last_ckpt_name

        complex_ckpt = torch.load(complex_ckpt, 'cpu')
        
        input = (complex_ckpt['conv1.weight'].data, complex_ckpt['conv2.weight'].data,
                complex_ckpt['bn1.weight'].data, complex_ckpt['bn1.bias'].data, complex_ckpt['bn1.running_mean'].data, complex_ckpt['bn1.running_var'].data,
                complex_ckpt['bn2.weight'].data, complex_ckpt['bn2.bias'].data, complex_ckpt['bn2.running_mean'].data, complex_ckpt['bn2.running_var'].data)
        
        return begin_ckpt, input, last_ckpt

    def __len__(self):
        return len(self.begin_ckpts)