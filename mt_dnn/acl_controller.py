import sys
import json
import torch
import random
import copy
import numpy as np
from shutil import copyfile
import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class BasicController():
    def __init__(self, n_task, dataset_names=None, dataset_sizes=None, max_cnt=50, batch_size=None, rebatch_size=None, max_step=1000, tensorboard=None, log_filename=None):
        self.n_task = n_task
        self.buffer = [myQueue(max_cnt, batch_size=batch_size, rebatch_size=rebatch_size) for _ in range(n_task)]
        self.task_losses = [buff.smooth_loss for buff in self.buffer]
        self.cur_step = 0
        self.cur_epoch = 0
        self.max_step = max_step
        self.max_cnt = max_cnt
        self.tensorboard = tensorboard
        self.sampled_cnt = [0] * n_task
        self.chosen_cnt = [0] * n_task
        self.trained_cnt = [0] * n_task
        self.dataset_names = dataset_names
        # the number of samples
        self.dataset_sizes = dataset_sizes
        # the number of batchs
        self.lengths = None
        # print("dataset sizes, ", dataset_sizes)
        assert batch_size is not None, "batch size should not be None"
        self.batch_size = batch_size
        self.rebatch_size = rebatch_size
        self.name_dict = {}
        self.scaled_dict = {}
        if dataset_names is None:
            self.dataset_names = ["task_%02d"%i for i in range(n_task)]
        
        for i, (x, y) in enumerate(zip(self.dataset_names, self.chosen_cnt)):
            self.name_dict[x] = y
            if self.dataset_sizes:
                self.scaled_dict[x] = self.trained_cnt[i] * 1.0 / self.dataset_sizes[i]
        
        
        self.global_step = 0
        self.log_file = None
        if log_filename:
            self.log_file = open(log_filename, "w")
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def insert(self, task_id, data, loss):
        self.buffer[task_id].append((data, loss))
        self.task_losses[task_id].add(loss)
        self.global_step += 1
        self.sampled_cnt[task_id] += 1
    
    def recal_loss(self, model):
        losses = []
        new_losses = []
        for buffer in self.buffer:
            losses.append([])
            new_losses.append([])
            for i, (data, loss) in enumerate(buffer.data):
                batch_meta, batch_data = data
                # print("loss",loss)
                losses[-1].append(loss)
                new_loss = model.calculate_loss(batch_meta, batch_data).item()
                buffer.data[i] = (data, new_loss)
                # print("new_loss",new_loss)
                new_losses[-1].append(new_loss)
        if self.log_file and random.random() < 0.2:
            line = {'losses': losses, 'new_losses': new_losses}
            self.write_file(**line)
        
    def calculate_loss(self):
        losses = []
        for i, data in enumerate(self.buffer):
            loss = data.calculate_loss()
            losses.append(loss)
        return losses
    
    def set_lengths(self, lengths):
        self.lengths = copy.deepcopy(lengths)

    def initalization(self):
        self.max_step = self.__len__()
        self.cur_step = 0
        for data in self.buffer:
            data.empty()
    
    def get_loss(self):
        avg_loss = ["%.5f"%task_loss.get_loss() for task_loss in self.task_losses]
        out_loss = ["%.5f"%task_loss.get_out_loss() for task_loss in self.task_losses]
        loss_change = ["%.5f"%task_loss.get_loss_change() for task_loss in self.task_losses]
        min_loss = ["%.5f"%task_loss.min_loss for task_loss in self.task_losses]
        min_out_loss = ["%.5f"%task_loss.min_out_loss for task_loss in self.task_losses]
        return avg_loss, out_loss, loss_change, min_loss, min_out_loss

    def __len__(self):
        raise NotImplementedError("__length__ function not implemented!")

    def get_task_id(self):
        raise NotImplementedError("Getting task index function must be overwritten!")
    
    def choose_task_index_to_update(self):
        raise NotImplementedError("Choosing update task index function must be overwritten!")
    
    def update_chosen_cnt(self, task_id):
        # tensorboard record chosen count
        self.chosen_cnt[task_id] += 1
        self.trained_cnt[task_id] += self.buffer[task_id].sample_cnt()
        
        task_name = self.dataset_names[task_id]

        self.name_dict[task_name] = self.chosen_cnt[task_id]
        if self.dataset_sizes:
            self.scaled_dict[task_name] = self.trained_cnt[task_id]*1.0 / self.dataset_sizes[task_id]        
        
        loss = self.buffer[task_id].calculate_loss()
        ema_loss = self.task_losses[task_id].get_loss()

        # tensorboard
        if self.tensorboard and random.random() < 0.3:
            self.tensorboard.add_scalars('train/chosen', self.name_dict, global_step=self.global_step)
            self.tensorboard.add_scalar('train/loss_%s'%task_name, loss, global_step=self.global_step)
            self.tensorboard.add_scalars('train/task_loss', {task_name: loss}, global_step=self.global_step)
            if self.dataset_sizes:
                self.tensorboard.add_scalars('train/scaled_chosen', self.scaled_dict, global_step=self.global_step)
        
        if self.log_file:
            line = {'task': task_name, 'chosen': self.name_dict[task_name]}
            if self.dataset_sizes:
                line['scaled_chosen'] = self.scaled_dict[task_name]
            self.write_file(**line)
            line = {'task': task_name, 'loss': loss, "ema_loss": ema_loss}
            self.write_file(**line)
            line = {'task': task_name, 'sampled_cnt': self.sampled_cnt[task_id], "pop_cnt": self.buffer[task_id].pop_cnt}
            line['valid_cnt'] = line['sampled_cnt'] - line['pop_cnt']
            self.write_file(**line)                     

    def step(self, model):
        arg_task_index = self.choose_task_index_to_update()
        if arg_task_index is not None:
            self.update_chosen_cnt(arg_task_index)
            # update task
            for data in self.buffer[arg_task_index]:
                batch_meta, batch_data = data
                model.update(batch_meta, batch_data)
            self.buffer[arg_task_index].empty()
            
    
    def write_file(self, **kwargs):
        line = {"global_step": self.global_step, "epoch": self.epoch, "cur_step": self.cur_step}
        line.update(kwargs)
        self.log_file.write("%s\n" % json.dumps(line))

    def summary(self):
        st = "Current Step {0} / {1}  ({2:.2%})\n".format(self.cur_step, self.max_step, self.cur_step*1.0/self.max_step)
        avg_loss, out_loss, loss_change, min_loss, min_out_loss = self.get_loss()
        st += 'List of Task Names: {}\n'.format(", ".join(self.dataset_names))
        if self.cur_step == 0:
            if self.dataset_sizes is not None:
                dataset_sizes = ["%d"%v for v in self.dataset_sizes]
                st += 'List of dataset size {}\n'.format(", ".join(dataset_sizes))
                total_size = sum(self.dataset_sizes)
                dataset_sizes = ["%.4f"%(v*1.0/total_size) for v in self.dataset_sizes]
                st += 'List of dataset percentage {}\n'.format(", ".join(dataset_sizes))
        loss_values = ["%.6f" % v for v in self.calculate_loss()]
        st += 'List of current loss {}\n'.format(", ".join(loss_values))
        st += 'List of average smoothing loss {}\n'.format(", ".join(avg_loss))
        st += 'List of out_loss {}\n'.format(", ".join(out_loss))
        st += 'List of loss_change {}\n'.format(", ".join(loss_change))
        st += 'List of min_loss {}\n'.format(", ".join(min_loss))
        st += 'List of min_out_loss {}\n'.format(", ".join(min_out_loss))
        chosen = ["%s:%d"%(k,v) for k, v in self.name_dict.items()]
        st += 'List of chosen times {}\n'.format(", ".join(chosen))
        chosen = ["%s:%.3f"%(k,v) for k, v in self.scaled_dict.items()]
        st += 'List of scaled chosen times {}\n'.format(", ".join(chosen))
        buffer_cnts = ["%d"% len(v) for v in self.buffer]
        st += 'List of buffer size {}\n'.format(", ".join(buffer_cnts))
        sampled_cnt = ["%d" % v  for v in self.sampled_cnt]
        st += 'List of sampled count {}\n'.format(", ".join(sampled_cnt))
        pop_cnt = ["%d" % queue.pop_cnt for queue in self.buffer]
        st += 'List of pop count {}\n'.format(", ".join(pop_cnt))

        return st

class BasicPhiController(BasicController):
    def __init__(self, n_task, phi=0.3, K=2, **kwargs):
        super().__init__(n_task=n_task, **kwargs)
        self.phi = phi
        self.task_index = -1
        self.length = 0
        self.cnt = 0
        self.sampled_cnt = [0] * self.n_task

        self.is_first_loop = True
        self.not_loop_total = K * self.n_task
    
    def initalization(self):
        super().initalization()
        self.task_index = -1
        self.is_first_loop = True
        self.cnt = 0
        self.sampled_cnt = [0] * self.n_task
    
    def __len__(self):
        return sum(self.lengths)
    
    def choose_task_index_to_update(self):       
        if self.cnt > self.not_loop_total:
            self.cnt = 0
            self.is_first_loop = True
            # option 1: current loss
            losses = self.calculate_loss()
            # option 2: current smoothing loss
            # losses = [task_loss.get_loss() for task_loss in self.task_losses]
            
            # print(losses)
            if random.random() < self.phi:
                arg_task_index = np.argmax(losses)
                # print("Choose max index, ", arg_task_index)
            else:
                p = np.array(losses)
                p /= p.sum()
                arg_task_index = np.random.choice(list(range(self.n_task)), p=p, replace=False)
                # print("Choose random index, ", arg_task_index)
            self.cur_step += len(self.buffer[arg_task_index])
            return arg_task_index
        else:
            return None

class ACLController(BasicPhiController):
    def __init__(self, n_task, phi=0.3, K=2,  **kwargs):
        super().__init__(n_task=n_task, phi=phi, K=K, **kwargs)
        
        # exp3 algorithm
        self.w = [1.0] * n_task
        self.epsilon = 0.001
        self.threshold = self.max_cnt // 3
        self.pi = self.calculate_pi()
    
    def initalization(self):
        super().initalization()
        self.w = [1.0] * self.n_task
        self.pi = self.calculate_pi()
    
    def calculate_pi(self):
        pi = np.array(self.w)
        # if self.dataset_sizes:
        #     total_size = sum(self.dataset_sizes)
        #     dataset_sizes = [v*1.0/total_size for v in self.dataset_sizes]
        #     pi = pi/pi.sum()
        #     pi = pi * np.exp(np.array(dataset_sizes))
        return (1-self.epsilon)*pi/pi.sum() + self.epsilon/self.n_task

    def update_weight(self, task_id, reward):
        r = reward/self.pi[task_id]
        self.w[task_id] *= math.exp(self.epsilon * r / self.n_task)
        self.w[task_id] = min(self.w[task_id], 100)
    
    def get_task_id(self):
        if self.cur_step >= self.max_step:
            return None
        if self.is_first_loop:
            self.task_index += 1
            # if buffer surpass threshold, then pass the task
            while self.task_index < self.n_task and len(self.buffer[self.task_index]) > self.threshold:
                self.task_index += 1
            if self.task_index >= self.n_task:
                self.task_index = 0
                self.is_first_loop = False
        else:
            self.cnt += 1
            # losses = [task_loss.get_loss() for task_loss in self.task_losses]
            self.pi = self.calculate_pi()
            self.task_index = np.random.choice(list(range(self.n_task)), p=self.pi, replace=False)
            self.sampled_cnt[self.task_index] += 1
        
        return self.task_index
    
    def choose_task_index_to_update(self):       
        arg_task_index = super().choose_task_index_to_update()
        if arg_task_index is not None:
            
            # update weight
            max_sampled_cnt = max(self.sampled_cnt)
           
            # losses = [task_loss.get_loss() for task_loss in self.task_losses]
            # max_loss = max(losses)
            for i in range(self.n_task):
                # if max_loss > 0.0:
                #     importance = losses[i] / max_loss
                # else:
                #     importance = 0.0
                if arg_task_index == i:
                    self.update_weight(i, 1.0 * self.sampled_cnt[i] / max_sampled_cnt)
                else:
                    self.update_weight(i, -1.0 * self.sampled_cnt[i] / max_sampled_cnt)
            self.sampled_cnt = [0] * self.n_task
            
        return arg_task_index

    def summary(self):
        st = super().summary()
        w = ["%.4f" % v for v in self.w]
        st += 'List of self.w : {}\n'.format(", ".join(w))
        pi = ["%.4f" % v for v in self.pi]
        st += 'List of self.pi : {}\n'.format(", ".join(pi))
        return st

class ChangingController(ACLController):
    def __init__(self, n_task, **kwargs):
        super().__init__(n_task=n_task, **kwargs)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.phi = min(1.0, epoch*0.2)
        print("self phi", self.phi)


class RandomController(BasicController):
    def __init__(self, n_task, **kwargs):
        super().__init__(n_task=n_task, **kwargs)
        self.task_indices = []

    def __len__(self):
        return sum(self.lengths)
    
    def initalization(self):
        super().initalization()
        self.task_indices = []
        for i, length in enumerate(self.lengths):
            self.task_indices += [i] * length
        random.shuffle(self.task_indices)

    def get_task_id(self):
        if self.cur_step >= self.max_step:
            return None
        self.cur_step += 1
        return self.task_indices[self.cur_step - 1]
    
    def choose_task_index_to_update(self):
        choice = [i for i, buf in enumerate(self.buffer) if len(buf) > 0]
        if len(choice) == 1:
            return choice[0]
        elif len(choice) > 1:
            print("TWO MORE CHOICE!!!!!!!!!!!!")
            arg_task_index = np.random.choice(choice, replace=False)
            return arg_task_index
        else:
            return None

class myQueue():
    def __init__(self, n=3, batch_size=None, rebatch_size=None):
        self.n = n
        self.data = []
        self.batch_size = batch_size
        self.pop_cnt = 0
        self.sum_loss = 0.0
        self.smooth_loss = DataLoss()
        self.sum_loss = 0.0
        self.is_changed = False
        if batch_size and rebatch_size:
            self.rebatch_size = rebatch_size
            assert batch_size % rebatch_size == 0, "rebatch_size must devide batch_size"

    def append(self, item):
        if len(self.data) >= self.n:
            self.data.pop(0)
            self.pop_cnt += 1
        self.data.append(item)
        self.is_changed = True

    def __len__(self):
        return len(self.data)

    def empty(self):
        self.data = []
        self.is_changed = True
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def calculate_loss(self):
        if self.is_changed:
            self.sum_loss = 0.0
            cnt = len(self.data)
            # if cnt > 0:
            #     return max([loss for data, loss in self.data])
            # return 0.0  
            for i, (data, loss) in enumerate(self.data):
                self.sum_loss += loss
            if cnt > 0:
                self.sum_loss /= cnt
            self.is_changed = False
        return self.sum_loss
    
    def sample_cnt(self):
        return sum([len(data[0]['uids']) for data, loss in self.data])
        
    def __iter__(self): 
        if self.batch_size and self.rebatch_size and (self.batch_size != self.rebatch_size):
            for i, (data, loss) in enumerate(self.data):
                total_meta, total_data = data
                total_data = [torch.split(a_tensor, self.rebatch_size, dim=0) for a_tensor in total_data]
                uids = total_meta['uids']
                meta_list = []
                for j in range(0, self.batch_size, self.rebatch_size):
                    meta = copy.deepcopy(total_meta)
                    meta["uids"] = copy.deepcopy(uids[j:j+self.rebatch_size])
                    # print("batch_uids", meta["uids"])
                    meta_list.append(meta)
                for j, batch_data in enumerate(zip(*total_data)):
                    # print(meta_list[j])
                    # for tmp in batch_data:
                    #     print(tmp.size())
                    yield (meta_list[j], list(batch_data))
        else:
            for i, (data, loss) in enumerate(self.data):
                yield data


class DataLoss():
    def __init__(self, n=30, beta=0.8, beta_2=0.9):
        self.n = n
        self.beta = beta
        self.beta_2 = beta_2
        self.data = []
        self.loss = 0.0
        self.out_loss = None
        self.min_loss = 1000000.0
        self.min_out_loss = 1000000.0

    def add(self, value):
        if len(self.data) >= self.n:
            removed_data = self.data[0]
            self.data.pop(0)
            if self.out_loss is None:
                self.out_loss = removed_data
            else:
                self.out_loss = self.update(self.out_loss, removed_data, self.beta_2)

        self.data.append(value)
        self.loss = self.update(self.loss, value, self.beta)
        if self.loss < self.min_loss:
            self.min_loss = self.loss
        if self.out_loss is not None and self.out_loss < self.min_out_loss:
            self.min_out_loss = self.out_loss
    
    def update(self, avg, value, beta):
        return avg * beta + value * (1.0 - beta)

    def get_loss(self):
        return self.loss
    
    def get_out_loss(self):
        if self.out_loss is None:
            return 0
        else:
            return self.out_loss
    
    def get_loss_change(self):
        if self.out_loss is None:
            return 0.0
        else:
            return self.out_loss - self.loss
