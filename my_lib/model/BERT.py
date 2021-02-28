from . import ModelBase
from transformers import pipeline

class BERT(ModelBase):

    def __init__(self):
        pass

    def fit(self, X, y):
        print("already fit")
        return self

    def predict(self, X):
        """

        Args:
            X (str): string sentence

        Returns:

        """
        raise NotImplementedError()

    def score(self, X, y):
        raise NotImplementedError()

    # # Define the training parameters
    # num_samples = [1000, 5000, 10000, 100000, 500000]
    # epochs = 3
    # patience = 3
    # batch_size = 64
    # seq_len = 30
    # lr = 2e-5
    # clip = 1.0
    # log_level = logging.DEBUG
    #
    # # Run!
    # result_bert, model_trained_bert = train_cycles(train_df['text'], train_df['label'], vocab, num_samples, 'BERT',
    #                                                epochs, patience, batch_size, seq_len, lr, clip, log_level)
    #
    # # Save the model and show the result
    # torch.save(model_trained_lstm.state_dict(), output_dir + 'stocktwit_bert.dict')
    # result_bert