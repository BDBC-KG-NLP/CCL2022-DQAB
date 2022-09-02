import rocketqa


def train_dual_encoder(base_model, train_set):
    dual_encoder = rocketqa.load_model(model=base_model, use_cuda=True, device_id=0, batch_size=64)
    dual_encoder.train(train_set, 2, 'task2_de', save_steps=5000, learning_rate=1e-5,
                       log_folder='task2_dual_log')


def train_cross_encoder(base_model, train_set):
    cross_encoder = rocketqa.load_model(model=base_model, use_cuda=True, device_id=3, batch_size=64)
    cross_encoder.train(train_set, 10, 'task1_cross', save_steps=3000, learning_rate=3e-5, log_folder='task1_cross_log')


if __name__ == '__main__':
    train_dual_encoder('zh_dureader_de_v2', '../data/task2_dual.tsv')
    # train_cross_encoder('zh_dureader_ce_v2', '../data/train_task1_cross.tsv')
