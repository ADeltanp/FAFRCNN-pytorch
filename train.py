from transfer_train import transfer_train
from utils.config import opt


def train(**kwargs):
    opt._parse(kwargs)
    # best_path = source_train()
    best_path = 'F:\\Projects\\RtCV\\Projects\\Arc\\checkpoints\\fasterrcnn_11230744_0.18576796229338355'
    transfer_train(best_path)


if __name__ == '__main__':
    import fire
    fire.Fire()
    train()
