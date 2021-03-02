import models
from keras.callbacks import EarlyStopping


class Sentence_Similarity_Model(object):
    def __init__(self, model_name='global_local_share_model', max_len=200, vocab_size=10000, embedding_size=256,
                 lstm_units=128, cnn_filters=256, kernel_size=3, local_ratio=0.5, global_ratio=0.5,
                 batch_size=64, epochs=100, x_train_left=None, x_train_right=None, y_train=None,
                 x_test_left=None, x_test_right=None, y_test=None, pretrain_vec=None, exist_emb=False):
        self.model_name = model_name
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.local_ratio = local_ratio
        self.global_ratio = global_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.x_train_left = x_train_left
        self.x_train_right = x_train_right
        self.y_train = y_train
        self.x_test_left = x_test_left
        self.x_test_right = x_test_right
        self.y_test = y_test
        self.pretrain_vec = pretrain_vec
        self.exist_emb = exist_emb

    def train(self):
        if self.model_name == 'global_local_share_model':
            model = models.global_local_share_model(self, summary=True)
        elif self.model_name == 'global_model':
            model = models.global_model(self, summary=True)
        elif self.model_name == 'local_model':
            model = models.local_model(self, summary=True)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

        model.fit([self.x_train_left, self.x_train_right], [self.y_train], batch_size=64, epochs=100,
                  validation_split=0.1, callbacks=[es])
        test_loss, test_acc = model.evaluate([self.x_test_left, self.x_test_right], [self.y_test])

        print("TEST Loss : {:.6f}".format(test_loss))
        print("TEST ACC : {:.6f}".format(test_acc))
