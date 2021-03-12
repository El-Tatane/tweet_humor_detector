from . import ModelBase

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


class BERT(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", num_labels=4)

    def fit(self, path):

        data = tf.keras.preprocessing.text_dataset_from_directory(path, seed=123, batch_size=30000)

        for i in data.take(1):
            train_feat = i[0].numpy()
            train_lab = i[1].numpy()

        df_data = pd.DataFrame([train_feat, train_lab]).T
        df_data.columns = ["DATA_COLUMN", "LABEL_COLUMN"]
        df_data["DATA_COLUMN"] = df_data["DATA_COLUMN"].str.decode("utf-8")

        train_InputExamples = convert_data_to_examples(df_data, "DATA_COLUMN", "LABEL_COLUMN")

        train_data = convert_examples_to_tf_dataset(list(train_InputExamples), self.tokenizer)
        train_data = train_data.shuffle(100).batch(32).repeat(2)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

        self.model.fit(train_data, epochs=2)

        return self

    def predict(self, X):
        """

        Args:
            X (str or string list): string sentence

        Returns:

        """

        tf_batch = self.tokenizer(X, max_length=128, padding=True, truncation=True, return_tensors='tf')
        tf_outputs = self.model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)

        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        return label


    def score(self, X, y):

        y_pred = []
        for i, X_el in enumerate(X):
            if i % 100 == 0:
                print(i, end="")
            res: int = self.predict(X_el)[0]

            y_pred.append(res)
        y_pred = np.array(y_pred)

        return accuracy_score(list(y), list(y_pred))


    def save(self, filename):
        self.model.save(filename)


    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        return self



# useful function


def convert_data_to_examples(train, DATA_COLUMN='DATA_COLUMN', LABEL_COLUMN="LABEL_COLUMN"):
    train_InputExamples = train.apply(
        lambda x: InputExample(
            guid=None,  # Globally unique ID for bookkeeping, unused in this case
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN],
        ),
        axis=1,
    )

    return train_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )
