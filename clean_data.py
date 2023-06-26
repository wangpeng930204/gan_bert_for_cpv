import logging
import os

import dutch_words
import fasttext
import pandas as pd
import spacy
import wget

logger = logging.getLogger(__name__)


class CleanData:
    """Read input data (output from cpv.data_sources.LocalData in data/enriched)
    and clean it / perform further preprocessing. Writes output to data/preprocessed.

    Parameters
    ----------
    data_path : str (default="../data/enriched/data_tenderned.csv")
    feature_name : str(default="Korte beschrijving aanbesteding")
        Column / feature name used for NLP analysis
    """

    def __init__(self, data, feature_name: str = "Korte beschrijving aanbesteding"):
        self.data = data
        self.feature_name = feature_name
        self.nlp = spacy.load("nl_core_news_lg")

    def _remove_short_entries(
            self, data: pd.DataFrame, n_min_words: int = 10
    ) -> pd.DataFrame:
        """Remove all entries that have fewer than n_min_words words.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        n_min_words : int (default=10)
            Minimum number of words we want to accept for a given description

        Returns
        -------
        data : pd.DataFrame
        """
        logger.info(f"Removing entries shorter than {n_min_words} words.")
        print(self.feature_name)
        keep_ids = [
            i
            for i in range(len(data))
            if len(data[self.feature_name].iloc[i].split()) > n_min_words
        ]
        data = data.iloc[keep_ids]
        return data

    def _remove_foreign_language_entries(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove all entries that are classified as foreign languages (non Dutch)

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        data : pd.DataFrame
        """
        logger.info("Removing non-dutch entries.")

        language_model_path = "./data/misc/lid.176.bin"

        if not os.path.exists(language_model_path):
            site_url = (
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
            _ = wget.download(site_url, out=language_model_path)

        lang_detector = fasttext.load_model(language_model_path)

        predictions = lang_detector.predict(
            [x.replace("\n", "") for x in data.loc[:, self.feature_name]]
        )

        df_lang = pd.DataFrame(
            {
                "language": [p[0] for p in predictions[0]],
                "confidence": [c[0] for c in predictions[1]],
            }
        )

        data = data.loc[(df_lang["language"] == "__label__nl").values, :]

        return data

    def _remove_special_characters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove all special characters, only keep letters and numbers.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        data : pd.DataFrame
        """
        logger.info("Removing special characters (keeping letters and numbers).")
        data[self.feature_name] = data[self.feature_name].apply(
            lambda X: " ".join(
                [
                    "".join([char for char in word if char.isalnum()])
                    for word in X.split()
                ]
            )
        )
        return data

    def _remove_numbers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove all numbers.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        data : pd.DataFrame
        """
        logger.info("Removing all numbers.")
        data[self.feature_name] = data[self.feature_name].apply(
            lambda X: " ".join(
                " ".join(
                    [
                        "".join([char for char in word if char.isalpha()])
                        for word in X.split()
                    ]
                ).split()
            )
        )
        return data

    def _find_geopolitical_entities(self, doc: str) -> list:
        """Find all words tagged as geopolitical entities by spacy.

        Parameters
        ----------
        doc : str
            Text to search for GPEs

        Returns
        -------
        pruned_geopolitical_entities : list
            List of words tagged as GPEs
        """
        doc = self.nlp(doc)
        geopolitical_entities = []
        for ent in doc.ents:
            if ent.label_ == "GPE":
                geopolitical_entities.append(ent.text)

        nl_words = dutch_words.get_ranked()
        pruned_geopolitical_entities = []
        for gpe in geopolitical_entities:
            if gpe.lower() not in nl_words:
                pruned_geopolitical_entities.append(gpe)

        return pruned_geopolitical_entities

    def _remove_geographic_information(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove geographically specific information - we do not want to base any
        decisions on e.g. a city or a word like north or south.

        We accomplish this by tagging Geopolitical Entity with spacy and removing them

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        data : pd.DataFrame
        """
        logger.info("Removing geospatial words and locations.")
        data = data.copy()
        for idx, row in data.iterrows():
            geopolitical_entities = self._find_geopolitical_entities(
                row[self.feature_name]
            )
            pruned_doc = [
                word
                for word in row.loc[self.feature_name].split()
                if word not in geopolitical_entities
                   and word.replace(".", "") not in geopolitical_entities
            ]
            data.loc[idx, self.feature_name] = " ".join(pruned_doc)

        return data

    def clean_data(self) -> pd.DataFrame:
        """Wrapper for performing all implemented cleaning operations

        Returns
        -------
        pd.DataFrame
        """
        self.data = self._remove_short_entries(self.data)
        self.data = self._remove_foreign_language_entries(self.data)
        self.data = self._remove_special_characters(self.data)
        self.data = self._remove_geographic_information(self.data)

        return self.data

    def read_clean_data(self) -> pd.DataFrame:
        """Read clean data if in default location."""
        data = pd.read_csv(self.preproc_dir / "data_clean_tenderned.csv")
        return data

    def write_data(self, data: pd.DataFrame):
        """Wwrite the preprocessed data to the appropriate data folder."""
        save_path = self.preproc_dir / "data_clean_tenderned.csv"
        logger.info(f"Writing preprocessed data to {save_path}")
        data.to_csv(save_path, index=False)

    def read_and_write_data(self):
        # overwrite parent method to avoid accidents
        pass
