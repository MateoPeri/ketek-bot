import difflib
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def get_word_score(w1, w2):
    count = 0
    for c1, c2 in zip(w1, w2):
        if c1 == c2:
            count += 1
    acc = count * 2 / (len(w1) + len(w2))
    # print(w1, w2, acc)
    return acc


def get_word_score2(w1, w2):
    res = w1.replace(w2, '')
    count = len(w1) - len(res)
    acc = count * 2 / (len(w1) + len(w2))
    return acc


def compare_words(w1, w2):
    return max(get_word_score(w1, w2), get_word_score2(w1, w2))


def find_similar_words_in_list(w, lst):
    # if lst.index(w) != i
    return sorted([(i, e, compare_words(w, e)) for i, e in enumerate(lst)], key=lambda x: x[2], reverse=True)


class KetekChecker:
    def __init__(self, debug=False, **kwargs):
        self.debug = debug

        self.symmetry_error = float(kwargs.get('symmetry_error'))
        self.outlier_cost = float(kwargs.get('outlier_cost'))
        self.word_match_threshold = float(kwargs.get('word_match_threshold'))

        self.char_to_remove = ['.', ',', '!', '\'', '?', ';', ':', '*', '[', ']']
        self.char_only_one = [' ', '\n']
        self.lemmatizer = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer("english")

    def check_ketek(self, text):
        if len(text.split()) < 5:
            return False, 0
        # print('original', text)
        words = self.get_generics(self.clean_input(text)).split()
        # print('generic', words)
        _words, pairs, out = self.remove_outliers(words)

        l_words = [words] + [_words + out]
        scores = []
        for wds in l_words:
            halfA, halfB = split(wds)
            halfB = halfB[::-1]
            # print(wds)
            # print(halfA, halfB)
            scores.append(self.calc_score(halfA, halfB, pairs, out))

        ketek_score = max(scores)
        # print('scores', scores)
        # print('best score was', ketek_score, 'in iteration', scores.index(ketek_score))
        is_ketek = ketek_score > 0.5

        return is_ketek, ketek_score

    def clean_input(self, inp):
        inp = inp.lower()
        # only words and whitespace
        inp = re.sub(r'[^a-zA-Z\s]+', '', inp)
        for c in self.char_only_one:
            inp = re.sub(f'{c}+', ' ', inp)
        for c in self.char_to_remove:
            inp = inp.replace(c, '')

        newL = inp.split()
        acc = 0
        for i, c in enumerate(inp.split()):
            if '-' in c:
                # hay que usar otra cosa, no i
                # porque i es el index
                newL = insert_list_in_list(newL, c.split('-'), i + acc, True)
                acc += len(c.split('-')) - 1

        return ' '.join(newL)

    def get_generics(self, text):
        # pos_tagged = nltk.pos_tag(text)
        lem = [self.lemmatizer.lemmatize(w, pos=get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
        stem = [self.snowball_stemmer.stem(w) for w in nltk.word_tokenize(' '.join(lem))]
        final = ' '.join(stem)
        # print(final)
        return final

    def calc_score(self, half_a, half_b, pairs, out):
        n_words = len(half_a) + len(half_b)
        sm = difflib.SequenceMatcher(None, half_a, half_b)
        ratio = sm.ratio()

        symm_scores = [compare_words(w1, w2) for w1, w2 in zip(half_a, half_b)]
        s_score = sum(symm_scores) / len(symm_scores)
        s_score += self.symmetry_error
        pair_scores = [p[2] for p in pairs]
        p_score = sum(pair_scores) / int(n_words / 2)

        cust_score = (s_score + p_score) / 2
        outliers = (len(out) / n_words) * self.outlier_cost
        cust_score -= outliers
        ketek_score = (cust_score + ratio) / 2 if ratio > cust_score else cust_score
        ketek_score = min(1.0, ketek_score)

        if self.debug:
            print('Ratio:', ratio)
            print('Symmetry score:', s_score, 'Pair score:', p_score)
            print('Custom score:', cust_score)
            print('Outliers penalty', outliers)

        return ketek_score

    def remove_outliers(self, words):
        # iterar over all items
        # find the most similar to me, from a list that does not include me
        # ask the most similar if i'm their most similar
        # if so, take em out (both)
        # else, keep going
        to_keep = []
        pairs = []
        _w = words.copy()
        __w = words.copy()
        for w in words:
            # _w is a copy of the original, and the pool from where we take the words
            # _w1 and _w2 are copies of the pool, each without the word we are iterating over
            # TODO: REFACTOR
            if w not in _w:
                continue
            _w1 = _w.copy()
            _w2 = _w.copy()
            _w1.remove(w)
            m1 = find_similar_words_in_list(w, _w1)
            mid, _ = find_middle(words)
            # word is axis of symmetry, has no pair but it is not an outlier
            if len(m1) == 0 and words.index(w) == mid:
                idx = get_index(__w, w, copy=False)
                # print('found axis', w)
                to_keep.append((idx, w))
                _w.remove(w)
                continue

            _w2.remove(m1[0][1])
            m2 = find_similar_words_in_list(m1[0][1], _w2)

            # print('words for', f'"{w}"', m1)
            # print('words for', f'"{m1[0][1]}"', m2)
            # print('testing...', w, '=>', m1[0][1], m1[0][1], '=>', m2[0][1])
            # if we are the best match for our best match
            if w == m2[0][1] and m2[0][2] > self.word_match_threshold:  # TODO: make variable
                idx1 = get_index(__w, m1[0][1], copy=False)
                idx2 = get_index(__w, m2[0][1], copy=False)
                # print('www', __w)
                to_keep.append((idx1, m1[0][1]))
                to_keep.append((idx2, m2[0][1]))
                pairs.append((m1[0][1], m2[0][1], m1[0][2]))
                # _w.remove(w)
                _w.remove(m1[0][1])
                _w.remove(m2[0][1])

        # print('unsorted', to_keep)
        words_to_keep = [w[1] for w in sorted(to_keep, key=lambda x: x[0])]
        # print('keeping words', words_to_keep)

        out = [w for w in words if w not in words_to_keep]
        # print('outliers', out)

        return words_to_keep, pairs, out


def split(words):
    mid, _ = find_middle(words)
    if _:
        # print('even')
        halfA = words[:mid + 1]
        halfB = words[_:]
    else:
        # print('odd')
        halfA = words[:mid]
        halfB = words[mid + 1:]
    return halfA, halfB


def find_middle(lst: list):
    mid = float(len(lst)) / 2
    return (int(mid - .5), None) if len(lst) % 2 != 0 else (int(mid - 1), int(mid))


def get_index(words, w, copy=True):
    if copy:
        words = words.copy()
    idx = words.index(w) if w in words else None
    if idx is not None:
        # print('word', w, 'with index', idx)
        words[idx] = '-1'
    else:
        # print('word', w, 'not found in', words)
        pass
    return idx


def insert_list_in_list(l1: list, l2: list, i: int, replace=False):
    fin = i + 1 if replace else i
    return l1[:i] + l2 + l1[fin:]
