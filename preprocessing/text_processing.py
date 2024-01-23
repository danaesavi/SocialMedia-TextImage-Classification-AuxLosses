import regex
import logging
import csv
import html
import random
import pandas as pd
import sys

logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )

logger = logging.getLogger(__name__)

def prepare_data_key_mhp():
    def combine_text(df):
        """
        Combines tweet and image text into one column
        df: Dataframe which holds the data
        taken from : https://github.com/botelhoa/Dog_Whistle_Hate/
        """
        combined_text = []

        for row_num in range(len(df)):
            tweet_text = df.loc[row_num, "tweet_text"]
            image_text = df.loc[row_num, "img_text"]
            if type(image_text) == str:
                combined_text.append(tweet_text + image_text)
            else:
                combined_text.append(tweet_text)
        return combined_text
    
    raw_data_path = "../data/MHP/Data/{}/dog_whistle_{}.csv"
    splits = ["train","validation","test"]
    dfs = []
    for split in splits:
        split_str = split if split != "validation" else "dev"
        df = pd.read_csv(raw_data_path.format(split.title(),split_str), encoding='utf-8')
        df["text"] = combine_text(df)
        df["label"] = df["Primary_numeric_gt"].astype(int)
        df["tweet_id"] = df["image_number"].astype(str)
        df = df[["tweet_id","text","label"]]
        split_str = split if split != "validation" else "val"
        df["split"] = [split_str]*len(df)
        dfs.append(df)
    
    data = pd.concat(dfs)
    print("value counts", data.split.value_counts())
    # train 3998 | val 502 | test 500
    print("train value counts", data[data.split=="train"].label.value_counts())  
    print("label value counts (all)", data.label.value_counts())
    data.to_csv("../data/data_key_mhp.csv",index = False)
    print("../data/data_key_mhp.csv saved!")
    print(data.head())

def prepare_data_key_mvsa():
    raw_data_path = "../data/MVSA-Single/data/"
    labels_txt ="../data/MVSA-Single/valid_pairlist.txt"
    # read each id- multimodal label, read the text
    with open(labels_txt) as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        file_id, label, _, _ = line.split(",")
        with open(raw_data_path + file_id + ".txt", encoding="ISO-8859-1") as f:
            text = f.readlines()
        text = text[0].encode('utf-8').strip()
        data_list.append(
            {"tweet_id": file_id,
             "text":text,
             "label": label
            }
        )
    data = pd.DataFrame(data_list)
    # random split 80/10/10
    tr_idxs = random.sample(range(0, len(data)), int(len(data)*.8))
    dev_test = set(range(len(data))) - set(tr_idxs)
    dev_idxs = random.sample(list(dev_test), int(len(data)*.1))
    split = []
    for idx in range(len(data)):
        if idx in tr_idxs:
            split.append("train")
        elif idx in dev_idxs:
            split.append("val")
        else:
            split.append("test")
    data["split"] = split
    print("value counts", data.split.value_counts())
    # train 3608 | val  451 | test  452
    data.to_csv("../data/data_key_mvsa.csv",index = False)
    print("../data/data_key_mvsa.csv saved!")
    print(data.head())


def prepare_data_key_tir(raw_data_path="../data/textimage-data.csv", split_mode = "imgtxt"):
    lines = []
    with open(raw_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
                headers = row
                print(len(headers))
            else:
                line_count += 1
                
                if len(row)>9:
                    row = row[:2]+[", ".join(row[2:-6])]+row[-6:]
                
                lines.append(row)
        print(f'Processed {line_count} lines.')
   
    data = pd.DataFrame(lines,columns=[headers])
    data = data.rename(columns={"tweet":"text"}).reset_index()
    if split_mode == "random":
        # random split
        #idxs = random.sample(range(0, len(data)), int(len(data)*.8))
        #split = ["train" if idx in idxs else "test" for idx in range(len(data))]
         # random split 80/10/10
        tr_idxs = random.sample(range(0, len(data)), int(len(data)*.8))
        dev_test = set(range(len(data))) - set(tr_idxs)
        dev_idxs = random.sample(list(dev_test), int(len(data)*.1))
        split = []
        for idx in range(len(data)):
            if idx in tr_idxs:
                split.append("train")
            elif idx in dev_idxs:
                split.append("val")
            else:
                split.append("test")
        data["split"] = split
        print("value counts", data.split.value_counts())
        # train 3576 | val 447 | test  448
        data.to_csv("../data/data_key_imgtxt_random.csv",index = False)
        print("../data/data_key_imgtxt_random.csv saved!")
    else:
        # assign train and test splits
        with open('{}_train.txt'.format(split_mode)) as f:
            train_ids = f.readlines()
        train_ids = {x.strip() for x in train_ids}
        
        with open('{}_test.txt'.format(split_mode)) as f:
            test_ids = f.readlines()
        test_ids = {x.strip() for x in test_ids}
        # sanity check
        ltr = {"T"+str(x[0]).strip() for i,x in data[["tweet_id"]].iterrows()} & train_ids
        print("len train", len(ltr))
        lte = {"T"+str(x[0]).strip() for i,x in data[["tweet_id"]].iterrows()} & test_ids
        print("len test", len(lte))
   
        not_found, split = [], []
        for i,x in data.iterrows():
            tweet_id = "T"+str(x[1]).strip()            
            if tweet_id in train_ids:
                split.append("train")
            elif tweet_id in test_ids:
                split.append("test")
            else:
                not_found.append(tweet_id)
                
        print("len split", len(split))
        print("len not_found", len(not_found))
        if len(split) == len(data):
            data["split"] = split
            print("value counts", data.split.value_counts())
            data.to_csv("data_key_new.csv",index = False)
        else:
            print("train and test ids don't match")


if __name__ == "__main__":
    #prepare_data_key_tir(split_mode = "random")
    #prepare_data_key_mvsa()
    prepare_data_key_mhp()


class Tweet_Preprocessing():
    def __init__(self):

        self.tweetPreprocessor = TweetTokenizer()
        self.special_puncts = {"’": "'", "…": "..."}
        try:
            from emoji import demojize

            self.demojizer = demojize
        except ImportError:
            logger.warning(
                "emoji is not installed, thus not converting emoticons or emojis into text. Install emoji: pip3"
                " install emoji==0.6.0"
            )
            self.demojizer = None

    def normalizeTweet(self, tweet):
        """
        Normalize a raw Tweet
        """
        for punct in self.special_puncts:
            tweet = tweet.replace(punct, self.special_puncts[punct])

        tokens = self.tweetPreprocessor.tokenize(tweet)
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return " ".join(normTweet.split())


    def normalizeToken(self, token):
        """
        Normalize tokens in a Tweet
        """
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            if token in self.special_puncts:
                return self.special_puncts[token]
            if self.demojizer is not None:
                return self.demojizer(token)
            else:
                return token
        else:
            return token

######################################################################

# Natural Language Toolkit: Twitter Tokenizer
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Christopher Potts <cgpotts@stanford.edu>
#         Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
#         Pierpaolo Pantone <> (modifications)
# URL: http://nltk.org/
# For license information, see LICENSE.TXT
#


"""
Twitter-aware tokenizer, designed to be flexible and easy to adapt to new domains and tasks. The basic logic is this:
1. The tuple regex_strings defines a list of regular expression strings.
2. The regex_strings strings are put, in order, into a compiled regular expression object called word_re.
3. The tokenization is done by word_re.findall(s), where s is the user-supplied string, inside the tokenize() method of
   the class Tokenizer.
4. When instantiating Tokenizer objects, there is a single option: preserve_case. By default, it is set to True. If it
   is set to False, then the tokenizer will lowercase everything except for emoticons.
"""


######################################################################
#
# import regex  # https://github.com/nltk/nltk/issues/2409
# import html
#
######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most importantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# ToDo: Update with http://en.wikipedia.org/wiki/List_of_emoticons ?

# This particular element is used in a couple ways, so we define it
# with a name:
# docstyle-ignore
EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

# URL pattern due to John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715
# docstyle-ignore
URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
    (?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

# docstyle-ignore
# The components of the tokenizer:
REGEXPS = (
    URLS,
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # exchange
      [ *\-.\)]*
      \d{4}          # base
    )""",
    # ASCII Emoticons
    EMOTICONS,
    # HTML tags:
    r"""<[^>\s]+>""",
    # ASCII Arrows
    r"""[\-]+>|<[\-]+""",
    # Twitter username:
    r"""(?:@[\w_]+)""",
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # email addresses
    r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",
    # docstyle-ignore
    # Remaining word types:
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)

######################################################################
# This is the core tokenizing regex:

WORD_RE = regex.compile(r"""(%s)""" % "|".join(REGEXPS), regex.VERBOSE | regex.I | regex.UNICODE)

# WORD_RE performs poorly on these patterns:
HANG_RE = regex.compile(r"([^a-zA-Z0-9])\1{3,}")

# The emoticon string gets its own regex so that we can preserve case for
# them as needed:
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)

# These are for regularizing HTML entities to Unicode:
ENT_RE = regex.compile(r"&(#?(x?))([^&;\s]+);")


######################################################################
# Functions for converting html entities
######################################################################


def _str_to_unicode(text, encoding=None, errors="strict"):
    if encoding is None:
        encoding = "utf-8"
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text


def _replace_html_entities(text, keep=(), remove_illegal=True, encoding="utf-8"):
    """
    Remove entities from text by converting them to their corresponding unicode character.
    Args:
        text:
            A unicode string or a byte string encoded in the given *encoding* (which defaults to 'utf-8').
        keep (list):
            List of entity names which should not be replaced. This supports both numeric entities (`&#nnnn;` and
            `&#hhhh;`) and named entities (such as `&nbsp;` or `&gt;`).
        remove_illegal (bool):
            If `True`, entities that can't be converted are removed. Otherwise, entities that can't be converted are
            kept "as is".
    Returns: A unicode string with the entities removed.
    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py
        >>> from nltk.tokenize.casual import _replace_html_entities >>> _replace_html_entities(b'Price: &pound;100')
        'Price: \\xa3100' >>> print(_replace_html_entities(b'Price: &pound;100')) Price: £100 >>>
    """

    def _convert_entity(match):
        entity_body = match.group(3)
        if match.group(1):
            try:
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                # Numeric character references in the 80-9F range are typically
                # interpreted by browsers as representing the characters mapped
                # to bytes 80-9F in the Windows-1252 encoding. For more info
                # see: https://en.wikipedia.org/wiki/ISO/IEC_8859-1#Similar_character_sets
                if 0x80 <= number <= 0x9F:
                    return bytes((number,)).decode("cp1252")
            except ValueError:
                number = None
        else:
            if entity_body in keep:
                return match.group(0)
            else:
                number = html.entities.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass

        return "" if remove_illegal else match.group(0)

    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))


######################################################################

######################################################################


class TweetTokenizer:
    r"""
    Examples:
    ```python
    >>> # Tokenizer for tweets.
    >>> from nltk.tokenize import TweetTokenizer
    >>> tknzr = TweetTokenizer()
    >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    >>> tknzr.tokenize(s0)
    ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']
    >>> # Examples using *strip_handles* and *reduce_len parameters*:
    >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    >>> s1 = "@remy: This is waaaaayyyy too much for you!!!!!!"
    >>> tknzr.tokenize(s1)
    [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    ```"""

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        Args:
            text: str
        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if
        `preserve_case=False`
        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove username handles
        if self.strip_handles:
            text = remove_handles(text)
        # Normalize word lengthening
        if self.reduce_len:
            text = reduce_lengthening(text)
        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        # Tokenize:
        words = WORD_RE.findall(safe_text)
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = list(map((lambda x: x if EMOTICON_RE.search(x) else x.lower()), words))
        return words


######################################################################
# Normalization Functions
######################################################################


def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)


def remove_handles(text):
    """
    Remove Twitter username handles from text.
    """
    pattern = regex.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)"
    )
    # Substitute handles with ' ' to ensure that text on either side of removed handles are tokenized correctly
    return pattern.sub(" ", text)


######################################################################
# Tokenization Function
######################################################################


def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    """
    Convenience function for wrapping the tokenizer.
    """
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles).tokenize(
        text
    )
