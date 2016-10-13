import unidecode
import numpy as np

character_set = set()

def parse_file(file_name):
    data_set = []
    with open(file_name, 'r') as f:
        current_book = None
        for line in f:
            if line.startswith('_BOOK_TITLE_'):
                if current_book is not None: data_set.append(current_book)
                current_book = ""
            else:
                current_book += line
    return data_set

#train_books = parse_file("../datasets/CBTest/data/cbt_train.txt")
#validation_books = parse_file("../datasets/CBTest/data/cbt_valid.txt")
#test_books = parse_file("../datasets/CBTest/data/cbt_test.txt")[:1]

seq = ("aaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaab"
       "aaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaab")

seq2 = ("aaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")

train_books = [seq2]

# For testing the algorithm, lets just reduce the total paramters by removing
# upper case letters and accents
train_books = [unidecode.unidecode(book).lower() for book in train_books]

for book in train_books:
    for letter in book:
        character_set.add(letter)

char_indices = { character : c_idx for c_idx, character in enumerate(sorted(character_set)) }
index_chars = { c_idx : character for character, c_idx in char_indices.items() }

print(sorted(char_indices), len(character_set))

def logits(str, int_type):
    assert len(char_indices) <= np.iinfo(int_type).max
    return np.array(list((char_indices[char] if char != 0 else 0) for char in str), dtype=int_type)

def one_hot(str):
    ret = np.zeros([len(str), len(character_set)])
    for idx, char in enumerate(str):
        if char != 0: ret[idx, char_indices[char]] = 1
    return ret

def output_as_str(probabilities):
    return "".join([index_chars[np.argmax(vec)] for vec in probabilities])
