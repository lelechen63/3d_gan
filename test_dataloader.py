from torch.utils.data import DataLoader
from dataset import VaganDataset
import time

dataset = VaganDataset('/mnt/disk1/dat/lchen63/grid/data/pickle/', train=True)
data_loader = DataLoader(dataset, batch_size=12,
                              num_workers=12,
                              shuffle=True, drop_last=True)
data_loader = iter(data_loader)

print('dataloader created')

for _ in range(len(data_loader)):
    start_time = time.time()
    data_loader.next()
    print("--- %s seconds ---" % (time.time() - start_time))

