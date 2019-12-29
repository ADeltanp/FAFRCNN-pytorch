from source_train import source_train
from transfer_train import transfer_train
from model.frcnn.utils.config import opt


def train(**kwargs):
    opt._parse(kwargs)
    best_path = source_train()
    transfer_train(best_path)


if __name__ == '__main__':
    import fire
    fire.Fire()
