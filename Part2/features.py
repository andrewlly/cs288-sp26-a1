from collections import ChainMap
from typing import Callable, Dict, Set

import pandas as pd
import re


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}
    
# ... (imports and FeatureMap class remain the same) ...

# 1. Improved Sentiment with Negation Handling
class ComplexSentiment(FeatureMap):
    name = "sentiment"

    POS_WORDS = {
        "good","great","love","excellent","amazing","wonderful","best","superb","perfect",
        "funny","beautiful","liked","enjoy","fantastic","happy","entertaining","delightful",
        "terrific","brilliant","smart","moving","touching","compelling","engaging","impressive",
        "charming","captivating","enjoyable","effective","worth"
    }
    NEG_WORDS = {
        "bad","terrible","awful","worst","hate","boring","stupid","waste","sucks","horrible",
        "mess","dull","poor","disappointment","fail","annoying","weak","predictable",
        "derivative","overlong","pretentious","mediocre","pointless"
    }

    # keep small + robust: we’ll also detect “isn’t/wasn’t/don’t/…” by suffix
    NEGATION = {"not","no","never","nothing","neither","nowhere"}
    CONTRAST = {"but","however","though","although","yet"}

    @classmethod
    def _clean(cls, w: str) -> str:
        # strip punctuation around words; keep internal apostrophes for "don't"
        return re.sub(r"(^[^a-z0-9']+|[^a-z0-9']+$)", "", w.lower())

    @classmethod
    def _is_negator(cls, w: str) -> bool:
        # handle: "not", "never", "can't", "isn't", "didn't", plus bare "n't"
        if w in cls.NEGATION or w == "n't":
            return True
        return w.endswith("n't") or w in {"cant","cannot","wont","dont","didnt","isnt","wasnt","werent","shouldnt","wouldnt","couldnt"}

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        raw = text.lower().split()
        toks = [cls._clean(w) for w in raw]
        toks = [t for t in toks if t]  # remove empty after cleaning

        f: Dict[str, float] = {}

        # 1) sentiment counts (capped) + balance
        pos = sum(1 for t in toks if t in cls.POS_WORDS)
        neg = sum(1 for t in toks if t in cls.NEG_WORDS)
        f["pos"] = float(min(pos, 3))
        f["neg"] = float(min(neg, 3))
        f["pos_minus_neg"] = float(max(-3, min(3, pos - neg)))

        # 2) short negation scope: next 2 tokens only
        neg_scope = 0
        for t in toks:
            if cls._is_negator(t):
                f["has_negation"] = 1.0
                neg_scope = 2
                continue

            if neg_scope > 0:
                if t in cls.POS_WORDS:
                    f["negates_pos"] = 1.0   # "not good"
                if t in cls.NEG_WORDS:
                    f["negates_neg"] = 1.0   # "not bad"
                neg_scope -= 1

        # 3) contrast pivot: after "but/however/yet" matters more
        pivot = None
        for i, t in enumerate(toks):
            if t in cls.CONTRAST:
                pivot = i
        if pivot is not None and pivot + 1 < len(toks):
            after = toks[pivot+1:]
            pos_a = sum(1 for t in after if t in cls.POS_WORDS)
            neg_a = sum(1 for t in after if t in cls.NEG_WORDS)
            f["has_contrast"] = 1.0
            f["after_contrast_balance"] = float(max(-3, min(3, pos_a - neg_a)))

        # 4) exclamation as a tiny intensity cue (cheap + safe)
        if "!" in text:
            f["has_exclaim"] = 1.0

        return cls.prefix_with_name(f)


class Bigrams(FeatureMap):
    name = "bigrams"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        features = {}
        words = text.lower().split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            features[bigram] = 1.0
        return self.prefix_with_name(features)


class DomainFeatures(FeatureMap):
    name = "domain"

    TECH_WORDS = {"windows", "dos", "file", "ftp", "server", "drive", "disk", "scsi", "mac", "cpu", "memory", "chip"}
    RELIGION_WORDS = {"god", "jesus", "christ", "bible", "church", "christian", "religion", "faith"}
    
    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        features = {}
        tokens = text.lower().split()
        
        # Feature 1: Tech Jargon Count
        features["tech_count"] = float(sum(1 for t in tokens if t in cls.TECH_WORDS))
        
        # Feature 2: Religion Jargon Count
        features["religion_count"] = float(sum(1 for t in tokens if t in cls.RELIGION_WORDS))
        
        # Feature 3: 'From' or 'Subject' lines (often in newsgroups headers)
        features["is_reply"] = 1.0 if "re:" in text.lower() else 0.0
        
        # Feature 4: Contains an email address (looks for @ symbol inside a word)
        features["has_email"] = 1.0 if any("@" in t for t in tokens) else 0.0

        return cls.prefix_with_name(features)
    
class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        # TODO: implement this! Expected # of lines: <5
        
        words = text.lower().split()
        f = {}
        for word in words:
            if word not in self.STOP_WORDS: f[word] = 1.0
        return self.prefix_with_name(f)
        


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, ComplexSentiment, DomainFeatures, Bias, Bigrams]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
