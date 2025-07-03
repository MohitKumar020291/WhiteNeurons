from pipeline.loadAnnotedData import torchClassDataset
import os
from pipeline.loadAnnotedData.helper import CreateCollectionOfSegmentatedImages


def load_train_data(type_='train'):
    dataset = CreateCollectionOfSegmentatedImages(type_=type_)
    return dataset

def load_infer_data():
    ...


if __name__ == "__main__":
    load_train_data()