import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import net
import os
from absl import app, flags
import loader
import copy
import logging
import utils
from tensorboardX import SummaryWriter
import mixer

FLAGS = flags.FLAGS
flags.DEFINE_string('gpu_id', '0', help='gpu id')
flags.DEFINE_integer('num_epochs', 300, help='number of epochs')
flags.DEFINE_integer('log_step', 25, help='log step')

flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam'], help='')
flags.DEFINE_float('lr', 0.001, help='learning rate')
flags.DEFINE_float('wd', 0., help='wd')
flags.DEFINE_float('grad_clip', -1, help='')
flags.DEFINE_integer('warmup', 0, help='')
flags.DEFINE_integer('warmup_iters', 0, help='')
flags.DEFINE_integer('batch_size', 4096, help='')
flags.DEFINE_string('scheduler', 'cos', help='')

flags.DEFINE_enum('mask_target', 't', ['t', 's', 'both'], help='')
flags.DEFINE_float('prob', 0., help='mask prob')
flags.DEFINE_enum('noise_target', 't', ['t', 's', 'both'], help='')
flags.DEFINE_float('noise', 0., help='')

flags.DEFINE_string('ckpt_path', './ckpts/nobn', help='ckpt path')
flags.DEFINE_string('log_path', './logs/nobn', help='log path')
flags.DEFINE_string('train_path', '/data/nobn', help='train path')
flags.DEFINE_string('val_path', '/data/val', help='val path')

flags.DEFINE_integer('num_layers', 24, help='')
flags.DEFINE_integer('s_dim', 72, help='')
flags.DEFINE_integer('c_dim', 128, help='')
flags.DEFINE_integer('hidden_s_dim', 256, help='')
flags.DEFINE_integer('hidden_c_dim', 512, help='')
flags.DEFINE_float('dropout', 0., help='')

flags.DEFINE_integer('train_data_length', 300000, help='')
flags.DEFINE_integer('num_cycles', 1, help='')

device = torch.device('cuda:0')

def warmup_lr(step):
    return min(step, FLAGS.warmup_iters) / FLAGS.warmup_iters

def random_mask(inputs, prob):
    assert isinstance(inputs, list)
    new_inputs = []
    for index in range(len(inputs)):
        mask = torch.rand_like(inputs[index])
        mask = torch.where(mask > 1 - prob, 0, 1)
        new_inputs.append(torch.mul(inputs[index], mask))
    return new_inputs

def random_noise(inputs, noise_level):
    assert isinstance(inputs, list)
    new_inputs = []
    for index in range(len(inputs)):
        noise = torch.randn_like(inputs[index]) * noise_level
        new_inputs.append(inputs[index] + noise)
    return new_inputs

def cal_loss(outputs, targets):
    assert isinstance(outputs, tuple)
    assert isinstance(targets, list)
    assert len(outputs) == len(targets)
    loss = torch.FloatTensor([0.]).to(device)
    for i in range(len(outputs)):
        loss += F.mse_loss(outputs[i], targets[i])
    return loss / len(outputs)

def func_to_detach(inputs):
    assert isinstance(inputs, list)
    new_inputs = []
    for index in range(len(inputs)):
        new_inputs.append(inputs[index].to(device).detach())
    return new_inputs

def eval(Net, testloader):
    Net.eval()
    correct = 0.
    total = 0.
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = Net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    assert total == 10000
    return 100 * correct / total

def train_and_eval():
    assert FLAGS.warmup >= 0
    assert FLAGS.scheduler in ['cos', 'linear', 'constant'] or FLAGS.scheduler.startswith('reduce')
    FLAGS.log_path = os.path.join(FLAGS.log_path, 
                                  'Train' + str(FLAGS.num_epochs) + 'EpochsUsing' + str(FLAGS.train_data_length) + 'data',
                                  str(FLAGS.num_layers) + '_' + str(FLAGS.s_dim) + '_' + str(FLAGS.c_dim) + '_' + str(FLAGS.hidden_s_dim) + '_' + str(FLAGS.hidden_c_dim),
                                  FLAGS.optimizer + '_lr=' + str(FLAGS.lr) + '_wd=' + str(FLAGS.wd) + '_clip=' + str(FLAGS.grad_clip) + '_warmup=' + str(FLAGS.warmup) + '_sched=' + FLAGS.scheduler + '_bs=' + str(FLAGS.batch_size),
                                  FLAGS.mask_target + str(FLAGS.prob) + '_' + FLAGS.noise_target + str(FLAGS.noise) + '_' + str(FLAGS.dropout))
    FLAGS.ckpt_path = os.path.join(FLAGS.ckpt_path, 
                                  'Train' + str(FLAGS.num_epochs) + 'EpochsUsing' + str(FLAGS.train_data_length) + 'data',
                                  str(FLAGS.num_layers) + '_' + str(FLAGS.s_dim) + '_' + str(FLAGS.c_dim) + '_' + str(FLAGS.hidden_s_dim) + '_' + str(FLAGS.hidden_c_dim),
                                  FLAGS.optimizer + '_lr=' + str(FLAGS.lr) + '_wd=' + str(FLAGS.wd) + '_clip=' + str(FLAGS.grad_clip) + '_warmup=' + str(FLAGS.warmup) + '_sched=' + FLAGS.scheduler + '_bs=' + str(FLAGS.batch_size),
                                  FLAGS.mask_target + str(FLAGS.prob) + '_' + FLAGS.noise_target + str(FLAGS.noise) + '_' + str(FLAGS.dropout))
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    if not os.path.exists(FLAGS.ckpt_path):
        os.makedirs(FLAGS.ckpt_path)
    utils.set_logger(os.path.join(FLAGS.log_path, 'train.log'))
    writer = SummaryWriter(log_dir=FLAGS.log_path)
    with open(os.path.join(FLAGS.log_path, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
        
    test_transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=test_transform)
    assert len(testset) == 10000
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    BeginBlock = net.BeginBlock().to(device)
    BeginBlock.eval()
    LastBlock = net.LastBlock().to(device)
    LastBlock.eval()

    translation_train_set = loader.TrainDataset(base_path=FLAGS.train_path, num=FLAGS.train_data_length)
    translation_train_loader = torch.utils.data.DataLoader(translation_train_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    translation_val_set = loader.ValDataset(base_path=FLAGS.val_path, num=100)
    translation_val_loader = torch.utils.data.DataLoader(translation_val_set, batch_size=1, shuffle=False, pin_memory=True)

    best_ave_acc = 0
    best_acc_list = []
    best_ckpt = None

    with torch.no_grad():
        acc_list = []
        for BeginParameters, _, LastParameters in translation_val_loader:
            SimpleBlock = net.SimpleBlock_NoBn().to(device)
            BeginBlock.load_state_dict(torch.load(BeginParameters[0], map_location='cuda'))
            LastBlock.load_state_dict(torch.load(LastParameters[0], map_location='cuda'))
            SimpleNet = nn.Sequential(BeginBlock, SimpleBlock, LastBlock)
            acc_list.append(eval(SimpleNet, testloader))
        ave_acc = sum(acc_list) / len(acc_list)
        logging.info('Random Init, Mean Accuracy {:.2f}%, Min Accuracy {:.2f}%, Max Accuracy {:.2f}%'.format(ave_acc, min(acc_list), max(acc_list)))
        count_list = [0 for _ in range(20)]
        for idx in range(len(acc_list)):
            count_list[int(acc_list[idx] // 5)] += 1
        logging.info('Accuracy Distribution {}'.format(count_list))

    translation_net = mixer.MLPMixer(num_layers=FLAGS.num_layers, S_dim=FLAGS.s_dim, C_dim=FLAGS.c_dim, hidden_S_dim=FLAGS.hidden_s_dim, hidden_C_dim=FLAGS.hidden_c_dim, dropout=FLAGS.dropout).to(device)
    SimpleBlock = net.SimpleBlock_NoBn().to(device)
    
    if FLAGS.optimizer == 'sgd':
        optimizer = optim.SGD(translation_net.parameters(), lr=FLAGS.lr, momentum=0.9, weight_decay=FLAGS.wd)
    elif FLAGS.optimizer == 'adam':
        optimizer = optim.Adam(translation_net.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.wd)

    FLAGS.warmup_iters = FLAGS.warmup * len(translation_train_loader)
    if FLAGS.warmup > 0:
        scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    if FLAGS.scheduler == 'cos':
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(FLAGS.num_epochs - FLAGS.warmup) // FLAGS.num_cycles)
    elif FLAGS.scheduler == 'linear':
        scheduler2 = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=FLAGS.num_epochs - FLAGS.warmup)
    elif FLAGS.scheduler.startswith('reduce'):
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(FLAGS.scheduler.split('_')[1]), patience=2)
        
    for epoch in range(FLAGS.num_epochs):
        total_loss = 0.
        for inputs, targets in translation_train_loader:
            inputs = inputs[:2]
            if FLAGS.prob > 0:
                if FLAGS.mask_target == 't':
                    inputs = random_mask(inputs, FLAGS.prob)
                elif FLAGS.mask_target == 's':
                    targets = random_mask(targets, FLAGS.prob)
                elif FLAGS.mask_target == 'both':
                    inputs = random_mask(inputs, FLAGS.prob)
                    targets = random_mask(targets, FLAGS.prob)
                else:
                    raise ValueError
            if FLAGS.noise > 0:
                if FLAGS.noise_target == 't':
                    inputs = random_noise(inputs, FLAGS.noise)
                elif FLAGS.noise_target == 's':
                    targets = random_noise(targets, FLAGS.noise)
                elif FLAGS.noise_target == 'both':
                    inputs = random_noise(inputs, FLAGS.noise)
                    targets = random_noise(targets, FLAGS.noise)
                else:
                    raise ValueError
            inputs, targets = func_to_detach(inputs), func_to_detach(targets)
            outputs = translation_net(inputs)
            loss = cal_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            if FLAGS.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(translation_net.parameters(), FLAGS.grad_clip)
            optimizer.step()
            if FLAGS.warmup > 0 and epoch < FLAGS.warmup:
                scheduler1.step()
            total_loss += loss.item()
        if epoch >= FLAGS.warmup and FLAGS.scheduler != 'constant':
            if FLAGS.scheduler.startswith('reduce'):
                scheduler2.step(total_loss)
            else: 
                scheduler2.step()
        logging.info('Epoch {}, Loss: {:.6f}'.format(epoch, total_loss / len(translation_train_loader)))
        writer.add_scalar('Loss', total_loss / len(translation_train_loader), epoch)
        writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()

        if (epoch % FLAGS.log_step == 0 and epoch != 0) or epoch == FLAGS.num_epochs - 1:
            with torch.no_grad():
                translation_net.eval()
                acc_list = []
                for BeginParameters, ComplexParameters, LastParameters in translation_val_loader:
                    BeginBlock.load_state_dict(torch.load(BeginParameters[0], map_location='cuda'))
                    LastBlock.load_state_dict(torch.load(LastParameters[0], map_location='cuda'))
                    copy_begin_block_dict = copy.deepcopy(BeginBlock.state_dict())
                    copy_last_block_dict = copy.deepcopy(LastBlock.state_dict())
                    ComplexParameters = func_to_detach(ComplexParameters)[:2]
                    SimpleParameters = translation_net(ComplexParameters)
                    assert len(SimpleBlock.state_dict().keys()) == 2
                    assert SimpleBlock.conv1.weight.shape == SimpleParameters[0][0].shape
                    assert SimpleBlock.conv2.weight.shape == SimpleParameters[1][0].shape
                    SimpleBlock.conv1.weight        = torch.nn.Parameter(SimpleParameters[0][0])
                    SimpleBlock.conv2.weight        = torch.nn.Parameter(SimpleParameters[1][0])
                    SimpleNet = nn.Sequential(BeginBlock, SimpleBlock, LastBlock)
                    acc_list.append(eval(SimpleNet, testloader))
                    for key in BeginBlock.state_dict().keys():
                        assert torch.equal(BeginBlock.state_dict()[key], copy_begin_block_dict[key])
                    for key in LastBlock.state_dict().keys():
                        assert torch.equal(LastBlock.state_dict()[key], copy_last_block_dict[key])
                ave_acc = sum(acc_list) / len(acc_list)
                logging.info('Epoch {}, Mean Accuracy {:.2f}%, Min Accuracy {:.2f}%, Max Accuracy {:.2f}%'.format(epoch, ave_acc, min(acc_list), max(acc_list)))
                writer.add_scalar('Mean_Accuracy', ave_acc, epoch)
                writer.flush()
                if ave_acc > best_ave_acc:
                    best_ave_acc = ave_acc
                    best_acc_list = acc_list
                    logging.info('Update Best Mean Accuracy {:.2f}%'.format(best_ave_acc))
                    best_ckpt = copy.deepcopy(translation_net.state_dict())
                translation_net.train()
    logging.info('Best Mean Accuracy {:.2f}%'.format(best_ave_acc))
    count_list = [0 for _ in range(20)]
    for idx in range(len(best_acc_list)):
        count_list[int(best_acc_list[idx] // 5)] += 1
    logging.info('Accuracy Distribution {}'.format(count_list))
    torch.save(best_ckpt, os.path.join(FLAGS.ckpt_path, str(round(best_ave_acc, 2)) + '.pth'))
    writer.close()

def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    torch.backends.cudnn.benchmark = True
    train_and_eval()

if __name__ == '__main__':
    app.run(main)