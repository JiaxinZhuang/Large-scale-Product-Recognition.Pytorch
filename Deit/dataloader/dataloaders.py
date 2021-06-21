# encoding=utf-8
from dataloader.products import *
from dataloader.transforms import get_transform
from torch.utils.data import DataLoader
from options import opt
import pdb

from torch.utils.data.distributed import DistributedSampler

###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集

###################

#train_list = "./datasets/train.txt"
train_list = "/youtu-reid/ericxian/aliproduct/project/datasets/index_aliproduct_train" + opt.version + ".txt"
val_list = "/youtu-reid/ericxian/aliproduct/project/datasets/val.txt"

def make_weights_for_balanced_classes(images, labels):
    print("Calculating class weights for" + str(max(labels)+1) + "classes")
    nclasses = (max(labels)+1)
    count = [0] * (max(labels)+1)
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight

max_size = 128 if opt.debug else None  # debug模式时dataset的最大大小




# transforms
transform = get_transform(opt.transform)
train_transform = transform.train_transform
val_transform = transform.val_transform

# datasets和dataloaders
train_dataset = TrainValDataset(train_list, transforms=train_transform, max_size=max_size)

# calculate weights
#############
weights = make_weights_for_balanced_classes(train_dataset.im_names, train_dataset.labels)
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#############

#train_dataset = TFRecordDataset(train_transform, version="_v1")




#train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
#train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=sampler, num_workers=opt.workers, drop_last=True)


#val_dataset = TFRecordDataset(val_transform,split="val")
#val_dataset = TrainValDataset(val_list, transforms=val_transform, max_size=max_size)
#val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers//2)

if TEST_DATASET_HAS_OPEN:
    test_list = "./datasets/test.txt"  # 测试集

    test_dataset = TestDataset(test_list, transforms=val_transform, max_size=max_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

else:
    test_dataloader = None
