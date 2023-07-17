import re
import time
import pubchempy as pcp
import logging

log = logging.getLogger(__name__)


def ans2cid(answer: str, paragraph_id: str = None) -> int:
    cids = []
    for word in re.split(" |/", answer):
        for char in [".", ","]:
            if word.endswith(char):
                word = word[:-1]  # remove punctuation marks at the end of words
        if word.lower() not in [
            "and",
            "at",
            "as",
            "in",
            "is",
            "of",
            "out",
            "the",
            "was",
            "acts",
        ]:  # common words which are also in the database as molecules
            for char in ["⋅", "·"]:  # crytsla water not recognzed by pubchempy
                if char in word:
                    word = word.split(char)[0]
                    break
            try:
                cid = pcp.get_cids(word.strip())
                if len(cid) > 0:
                    cids.append([cid[0], word])
            except pcp.PubChemHTTPError as e:
                log.error(e)
    if len(cids) == 0:
        return None
    if len(cids) > 1:
        log.warn(f'Found more than one cid: {cids} in "{answer}" for {paragraph_id}')
    return cids[0][0]


def txt2cid(txt: str) -> list[int]:
    try:
        return pcp.get_cids(txt.strip())
    except pcp.PubChemHTTPError as e:
        log.warning(e)
        time.sleep(2)
        return txt2cid(txt)


def cid2syns(cid: int) -> list[str]:
    try:
        return pcp.Compound.from_cid(cid).synonyms
    except pcp.PubChemHTTPError as e:
        log.warning(e)
        time.sleep(2)
        return cid2syns(txt)
