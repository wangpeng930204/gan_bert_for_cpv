from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class ProcessData:
    def __init__(self, df, label_column, sentence_column, up_limit=500, down_limit=500, drop_limit=40,
                 test_per_label_num=30,
                 balanced_test=True):
        self.df = df
        self.label = label_column
        self.sentence_column = sentence_column
        self.up_limit = up_limit
        self.down_limit = down_limit
        self.test_num = test_per_label_num
        self.balanced_test = balanced_test
        self.drop_limit = drop_limit
        self.label_encoder = LabelEncoder()
        self.onehot = OneHotEncoder()
        self.num_class = 0

    def _replace_newline_strip(self):
        self.df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""], regex=True, inplace=True)

    def _remove_duplicate_sentences(self):
        print("Before removing duplicate sentence:", len(self.df))
        self.df = self.df.drop_duplicates(subset=[self.sentence_column], keep='last')
        print("After removing duplicate sentence:", len(self.df))

    def _remove_low_observation_data(self):
        print("Before drop we have :", len(self.df))
        feature_obs_counts = self.df[self.label].value_counts()
        left_obs_feature = feature_obs_counts[feature_obs_counts.values > self.drop_limit]
        self.df = self.df[self.df[self.label].isin(left_obs_feature.index)]
        print("After drop cpv with low observations we have :", len(self.df))

    def _split_train_test(self):
        if self.balanced_test:
            test_df = self.df.sample(frac=1).groupby(self.label, sort=False).head(self.test_num)
        else:
            test_df = self.df.sample(frac=0.1)
        train_df = self.df.drop(test_df.index)
        print(f"The number of total data is: ", len(self.df))
        print(f"Total number of test data is: ", len(test_df))
        print(f"Total number of train data is: ", len(train_df))
        return test_df, train_df

    def _split_sufficient_insufficient(self, train_df):
        label_vc = train_df[self.label].value_counts()
        up_vc = label_vc[label_vc.values > self.up_limit]
        down_vc = label_vc[label_vc.values < self.down_limit]
        up_df = train_df[train_df[self.label].isin(up_vc.index)]
        down_df = train_df[train_df[self.label].isin(down_vc.index)]
        return up_df, down_df

    def _init_label_encoder(self):
        all_labels = list(set(self.df.loc[:, self.label].values))
        integer_encoded = self.label_encoder.fit_transform(all_labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot = self.onehot.fit_transform(integer_encoded)
        self.num_class = len(all_labels)
        print("The num of class is ", self.num_class)

    def _sample_balanced_data(self, sample_num):
        self.df = self.df.sample(frac=1).groupby(self.label, sort=False).head(sample_num)

    def process(self):
        self._remove_duplicate_sentences()
        self._remove_low_observation_data()
        self._replace_newline_strip()
        self._sample_balanced_data(1500)
        self._init_label_encoder()
        test_df, train_df = self._split_train_test()
        suf_train, insuf_train = self._split_sufficient_insufficient(train_df)
        return test_df, suf_train, insuf_train, self.label_encoder, self.onehot
