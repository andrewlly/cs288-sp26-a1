from collections import ChainMap
from typing import Callable, Dict, Set
from collections import defaultdict
import pandas as pd
import re

TOKEN_RE = re.compile(r"[A-Za-z']+|[.!?]")

def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text)]



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
class SentimentLexicon(FeatureMap):
    name = "lex"

    POS_WORDS = {
        "good", "great", "love", "excellent", "amazing", "wonderful", "best", 
        "superb", "perfect", "funny", "beautiful", "liked", "enjoy", "fantastic", 
        "happy", "entertaining", "delightful", "terrific", "brilliant", "smart", 
        "moving", "touching", "compelling", "engaging", "impressive", "worth",
        "solid", "fresh", "masterpiece", "fun", "powerful", "engrossing", 
        "captures", "remarkable", "delivers", "thoughtful", "warm", "hilarious", 
        "rare", "honest", "rich", "fascinating", "intelligent", "satisfying", 
        "crafted", "charming", "heart", "strong", "sweet", "memorable", "creative"
    }
    
    NEG_WORDS = {
        "bad", "terrible", "awful", "worst", "hate", "boring", "stupid", "waste", 
        "sucks", "horrible", "mess", "dull", "poor", "disappointment", "fail", 
        "annoying", "weak", "predictable", "derivative", "overlong", "pretentious", 
        "mediocre", "pointless", "slow", "flat", "unfunny", "cliche", "garbage",
        "contrived", "tired", "ugly", "bland", "tedious", "lack", "lacks", 
        "loud", "ridiculous", "painful", "poorly", "messy", "suffers", "amateurish", 
        "laughable", "tiresome", "incoherent", "mindless", "dry", "problem"
    }

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = set(re.split(r'[^a-z0-9]+', text.lower()))
        
        pos_count = len(words & self.POS_WORDS)
        neg_count = len(words & self.NEG_WORDS)
        
        return self.prefix_with_name({
            "pos_score": float(pos_count),
            "neg_score": float(neg_count),
            "net_score": float(pos_count - neg_count),
            "has_sentiment": 1.0 if (pos_count > 0 or neg_count > 0) else 0.0
        })

class NegationInteraction(FeatureMap):
    name = "neg"
    NEGATORS = {"not","no","never","n't","neither","without","nowhere"}

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        toks = tokenize(text)
        f = {}
        scope = 0

        for t in toks:
            if t in {".","!","?"}:
                scope = 0
                continue
            if t in cls.NEGATORS or t.endswith("n't"):
                f["has_negation"] = 1.0
                scope = 3
                continue
            if scope > 0 and re.fullmatch(r"[a-z']+", t):
                f[f"NEG_{t}"] = 1.0
                scope -= 1

        return cls.prefix_with_name(f)

    
class ContrastFlow(FeatureMap):
    name = "con"
    CONTRAST_WORDS = {"but","however","although","yet","though"}

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        toks = tokenize(text)
        pivot = -1
        for i, t in enumerate(toks):
            if t in cls.CONTRAST_WORDS:
                pivot = i

        if pivot == -1:
            return cls.prefix_with_name({})

        after = [t for t in toks[pivot+1:] if t not in {".","!","?"}]
        pos_after = sum(1 for t in after if t in SentimentLexicon.POS_WORDS)
        neg_after = sum(1 for t in after if t in SentimentLexicon.NEG_WORDS)

        return cls.prefix_with_name({
            "has_contrast": 1.0,
            "after_balance": float(pos_after - neg_after),
        })



class Bigrams(FeatureMap):
    name = "bigrams"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        toks = [t for t in tokenize(text) if t not in {".","!","?"}]
        f = {}
        for i in range(len(toks) - 1):
            f[f"{toks[i]}_{toks[i+1]}"] = 1.0
        return cls.prefix_with_name(f)


class DomainFeatures(FeatureMap):
    name = "domain"

    TECH_WORDS = {"windows", "dos", "file", "ftp", "server", "drive", "disk", "scsi", "mac", "cpu", "memory", "chip"}
    RELIGION_WORDS = {"god", "jesus", "christ", "bible", "church", "christian", "religion", "faith"}
    
    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        features = {}
        tokens = text.lower().split()
        
        features["tech_count"] = float(sum(1 for t in tokens if t in cls.TECH_WORDS))
        
        features["religion_count"] = float(sum(1 for t in tokens if t in cls.RELIGION_WORDS))

        features["is_reply"] = 1.0 if "re:" in text.lower() else 0.0
        
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

class BagOfWordsCounts(FeatureMap):
    name = "bowc"
    STOP_WORDS = set(pd.read_csv("sentiment_stopword.txt", header=None)[0])

    TOKEN_RE = re.compile(r"[A-Za-z']+")

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = [t.lower() for t in self.TOKEN_RE.findall(text)]
        f = defaultdict(float)
        for tok in tokens:
            if tok not in self.STOP_WORDS:
                f[tok] += 1.0
        return self.prefix_with_name(dict(f))


class StylisticFeature(FeatureMap):
    name = "style"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        f = {}
        if "!" in text:
            f["has_exclaim"] = 1.0
            
        words = text.split()
        caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        if caps_count > 0:
            f["has_caps"] = 1.0
            
        return cls.prefix_with_name(f)

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


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, SentimentLexicon, 
                                           DomainFeatures, Bigrams, BagOfWordsCounts, 
                                           NegationInteraction, ContrastFlow, StylisticFeature]}


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
