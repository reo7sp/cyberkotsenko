import json
import logging
import os
import pickle
import random
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Tuple

import gensim
import nltk
import numpy as np
from kts_linguistics.string_transforms.basic_normalize import BasicNormalizeByWordTransform
from kts_linguistics.string_transforms.transform_pipeline import TransformPipeline
from kts_linguistics.string_transforms.utility_transforms import TokenizeTransform, JoinTransform, FuncTransform, \
    FuncByWordTransform
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

_logger = logging.getLogger(__name__)

_filterer = None
_DEFAULT_TRAIN_JSON_PATH = os.path.join(os.path.dirname(__file__), 'simple_zadumchik_data', 'train.json')
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'simple_zadumchik_data', 'model.bin')
_DEFAULT_QUOTES_PATH = os.path.join(os.path.dirname(__file__), 'simple_zadumchik_data', 'quotes.txt')
_DEFAULT_QUOTES2_PATH = os.path.join(os.path.dirname(__file__), 'simple_zadumchik_data', 'quotes2.txt')
_DEFAULT_MORPH_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'simple_zadumchik_data', 'morph_cache.pkl')
_FAIL_QUOTE = 'Хорошего дня, Александр'


def make_filterer_base() -> TransformPipeline:
    global _filterer

    if _filterer is not None:
        return _filterer

    morph = MorphAnalyzer()

    nltk.download('stopwords')
    nltk.download('punkt')
    stopwords_ = stopwords.words('russian')

    morph_cache = {}
    morph_cache_dirty = False

    if os.path.isfile(_DEFAULT_MORPH_CACHE_PATH):
        with open(_DEFAULT_MORPH_CACHE_PATH, mode='rb') as f:
            morph_cache = pickle.load(f)

    # https://github.com/akutuzov/webvectors/blob/master/preprocessing/rusvectores_tutorial.ipynb
    # https://yandex.ru/dev/mystem/doc/grammemes-values-docpage/
    # https://pymorphy2.readthedocs.io/en/latest/user/grammemes.html#grammeme-docs
    def morph_normalize(w):
        nonlocal morph_cache_dirty

        if w == '':
            return None

        if w not in morph_cache:
            d = morph.parse(w)[0]

            if d.normal_form in stopwords_:
                morph_cache[w] = None
                return None

            pos = d.tag.POS
            if pos == 'ADJF' or pos == 'ADJF': pos = 'A'
            elif pos == 'ADVB': pos = 'ADV'
            elif pos == 'NUMR': pos = 'NUM'
            elif pos == 'PRCL': pos = 'PART'
            elif pos == 'PREP': pos = 'PR'
            elif pos == 'NOUN': pos = 'S'
            elif pos == 'NPRO': pos = 'SPRO'
            elif pos == 'VERB': pos = 'V'
            else: pos = 'X'

            morph_cache[w] = f'{d.normal_form}_{pos}'
            morph_cache_dirty = True

        return morph_cache[w]

    counter = 0

    def checkpoint(w):
        nonlocal counter, morph_cache_dirty

        if counter % 250000 == 0 and morph_cache_dirty:
            _logger.debug(f'filtered word counter: {counter}')
            with open(_DEFAULT_MORPH_CACHE_PATH, mode='wb') as f:
                pickle.dump(morph_cache, f)
            morph_cache_dirty = False

        counter = counter + 1
        return w

    pipeline = TransformPipeline()
    _filterer = pipeline
    pipeline.add_transform(TokenizeTransform())
    pipeline.add_transform(BasicNormalizeByWordTransform())
    pipeline.add_transform(FuncByWordTransform(morph_normalize))
    pipeline.add_transform(FuncTransform(lambda ws: [w for w in ws if w]))
    pipeline.add_transform(FuncByWordTransform(checkpoint))
    return pipeline


def make_filterer_join() -> TransformPipeline:
    pipeline = make_filterer_base().copy()
    pipeline.add_transform(JoinTransform())
    return pipeline


class Sentiment(Enum):
    POSITIVE = 'positive'
    NEUTRAL = 'neutral'
    NEGATIVE = 'negative'

    def opposite(self):
        if self == self.POSITIVE:
            return self.NEGATIVE
        elif self == self.NEGATIVE:
            return self.POSITIVE
        return random.choices([self.POSITIVE, self.NEGATIVE], weights=[0.5, 0.5])[0]


# https://www.kaggle.com/ziliwang/baseline-upsampling-balanced-softmax-regression
class SentimentClassifier:
    def __init__(self, filt: TransformPipeline):
        self._vect = None
        self._regr = None
        self._filt = filt

    @classmethod
    def make_default(cls):
        with open(_DEFAULT_TRAIN_JSON_PATH, mode='r') as f:
            j = json.load(f)
            s = cls(filt=make_filterer_join())
            s.fit(j)
            return s

    def fit(self, train):
        _logger.info(f'fit, {len(train)} examples')

        train = [{**d, 'text': t} for d in train for t in d['text'].replace('\\n', '\n').split() if t]
        _logger.info(f'fit, accually {len(train)} examples')

        _logger.info('fit: vect')
        self._vect = CountVectorizer()
        self._vect.fit([self._filt.transform(d['text']) for d in train])

        _logger.info('fit: regr')
        train_ = defaultdict(list)
        for e in train:
            train_[e['sentiment']].append(e['text'])
        self._regr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        btrain = self._upsampling_align(train_)
        train_x = [j for i in sorted(btrain.keys()) for j in btrain[i]]
        train_y = [i for i in sorted(btrain.keys()) for j in btrain[i]]
        self._regr.fit(self._vect.transform(train_x), train_y)

        _logger.info('fit done')

    def predict(self, test: List[str]) -> List[Sentiment]:
        if len(test) == 0:
            return []
        r = self._regr.predict(self._vect.transform([self._filt.transform(s) for s in test]))
        return [Sentiment(it) for it in r]

    @staticmethod
    def _upsampling_align(some_dict, random_state=2018):
        rand = np.random.RandomState(random_state)
        upper = max([len(some_dict[l]) for l in some_dict])
        tmp = {}
        for l in some_dict:
            if len(some_dict[l]) < upper:
                repeat_time = int(upper / len(some_dict[l]))
                remainder = upper % len(some_dict[l])
                _tmp = some_dict[l].copy()
                rand.shuffle(_tmp)
                tmp[l] = some_dict[l] * repeat_time + _tmp[:remainder]
                rand.shuffle(tmp[l])
            else:
                tmp[l] = some_dict[l]
        return tmp


class SentenceType(Enum):
    REGULAR = 1
    QUESTION = 2


class SentenceClassifier:
    def predict(self, test: str) -> SentenceType:
        if '?' in test:
            return SentenceType.QUESTION
        else:
            return SentenceType.REGULAR


# https://github.com/akutuzov/webvectors/blob/master/preprocessing/rusvectores_tutorial.ipynb
class TextSimilarityRegressor:

    def __init__(self, filt: TransformPipeline):
        self._model = None
        self._filt = filt
        self._cache: Dict[Tuple[str, str], float] = {}

    @classmethod
    def make_default(cls):
        s = cls(filt=make_filterer_base())
        s.fit(_DEFAULT_MODEL_PATH)
        return s

    def fit(self, filename: str) -> None:
        self._model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict(self, a: str, b: str) -> float:
        res = 0
        aws = self._filt.transform(a)
        for aw in aws:
            _logger.debug(aws)
            for bw in self._filt.transform(b):
                key = min((aw, bw), (bw, aw))
                if key not in self._cache:
                    try:
                        self._cache[key] = self._model.similarity(aw, bw)
                    except KeyError:
                        _logger.warning('model error', exc_info=True)
                        self._cache[key] = 0
                res = max(res, self._cache[key] + random.random())
        return res


class SimpleZadumchikGenerator:
    def __init__(self,
                 sent_classifier: SentimentClassifier,
                 sentc_classifier: SentenceClassifier,
                 sim_regressor: TextSimilarityRegressor):
        self._sent_classifier = sent_classifier
        self._sentc_classifier = sentc_classifier
        self._sim_regressor = sim_regressor
        self._quotes_sents: Dict[Tuple[SentenceType, Sentiment], List[str]] = defaultdict(list)

    @classmethod
    def make_default(cls):
        s = cls(sent_classifier=SentimentClassifier.make_default(),
                sentc_classifier=SentenceClassifier(),
                sim_regressor=TextSimilarityRegressor.make_default())

        with open(_DEFAULT_QUOTES_PATH, mode='r') as f:
            s.fit_quotes(f.readlines())
        with open(_DEFAULT_QUOTES2_PATH, mode='r') as f:
            s.fit_quotes(f.readlines())

        return s

    def fit_quotes(self, quotes: List[str]) -> None:
        _logger.info('fit')

        for q, s in zip(quotes, self._sent_classifier.predict(quotes)):
            t = self._sentc_classifier.predict(q)
            self._quotes_sents[(t, s)].append(q)

        _logger.info(f'fit done, {sorted({(k, len(v)) for k, v in self._quotes_sents.items()}, key=lambda it: it[1])}')

    def generate(self, msg: str) -> str:
        msg_type = self._sentc_classifier.predict(msg)
        msg_sent = self._sent_classifier.predict([msg])[0]

        _logger.info(f'msg_type: {msg_type}, msg_sent: {msg_sent}')

        ans_type, ans_sent = self._rule_sentiment(msg_type, msg_sent)
        avail_quotes = self._quotes_sents[(ans_type, ans_sent)]

        _logger.info(f'ans_type: {ans_type}, ans_sent: {ans_sent}, avail_quotes: {len(avail_quotes)}')

        if len(avail_quotes) == 0:
            ans = _FAIL_QUOTE
        else:
            ans = max(avail_quotes, key=lambda q: self._sim_regressor.predict(msg, q))

        if ans[-1].isalnum():
            ans += '.'

        return ans

    @staticmethod
    def _rule_sentiment(src_type: SentenceType, src_sent: Sentiment) -> Tuple[SentenceType, Sentiment]:
        if src_type == SentenceType.REGULAR:
            dst_type = SentenceType.QUESTION
            dst_sent = src_sent.opposite()

        elif src_type == SentenceType.QUESTION:
            dst_type = random.choices([SentenceType.REGULAR, SentenceType.QUESTION], weights=[0.7, 0.3])[0]
            dst_sent = src_sent

        return dst_type, dst_sent
