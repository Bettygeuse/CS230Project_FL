import torch
from torch.utils.data import random_split, TensorDataset, DataLoader
import numpy as np

def get_dataset(dataset):
    samples = np.load(dataset)
    num_samples = len(samples)

    train_samples = samples[:-num_samples//5]
    test_samples = samples[-num_samples//5:]

    return train_samples, test_samples
    

    

def prepare_dataset(datasets, num_partitions: int, batch_size: int, val_ratio: float = 0.1, is_uniform=True):
    train_sample_sets = []
    test_sample_sets = []
    for dataset in datasets:
        train_samples, test_samples = get_dataset(dataset)
        train_sample_sets.append(train_samples)
        test_sample_sets.append(test_samples)
    

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    split_trainsets = []
    if is_uniform:
        for partition_idx in range(num_partitions):
            trainset_parts = []
            for trainset in train_sample_sets:
                num_samples = len(trainset) // num_partitions
                trainset_parts.append(trainset[partition_idx * num_samples : (partition_idx + 1) * num_samples, :, :])
            ret_trainset = np.concatenate(trainset_parts)

            split_trainsets.append(ret_trainset)


    else:
        concat_trainset = np.concatenate(train_sample_sets, axis=0)
        split_trainsets = np.array_split(concat_trainset, num_partitions)

    for set in split_trainsets:
        print(set.shape)

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for npsamples in split_trainsets:
        trainset_ = TensorDataset(
            torch.tensor(npsamples[:,:-10,:]).type(torch.FloatTensor),
            torch.tensor(npsamples[:,-10:,:3]).type(torch.FloatTensor)
        )

        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # We leave the test set intact (i.e. we don't partition it)
    # This test set will be left on the server side and we'll be used to evaluate the
    # performance of the global model after each round.
    # Please note that a more realistic setting would instead use a validation set on the server for
    # this purpose and only use the testset after the final round.
    # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
    # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
    # in main.py above the strategy definition for more details on this)
    npTestSamples = np.concatenate(test_sample_sets, axis=0)
    testset = TensorDataset(
        torch.tensor(npTestSamples[:,:-10,:]).type(torch.FloatTensor),
        torch.tensor(npTestSamples[:,-10:,:3]).type(torch.FloatTensor)
    )
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader

if __name__ == "__main__":
    prepare_dataset(["datasets/SumoSimulation/type1_data.npy", 
                    "datasets/SumoSimulation/type2_data.npy",
                    "datasets/SumoSimulation/type3_data.npy"], 5, 128, is_uniform=False)