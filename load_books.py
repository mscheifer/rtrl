

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
                for character in line:
                    character_set.add(character)
                current_book += line
    return data_set

train_books = parse_file("./datasets/CBTest/data/cbt_train.txt")
#validation_books = parse_file("./datasets/CBTest/data/cbt_valid.txt")
#test_books = parse_file("./datasets/CBTest/data/cbt_test.txt")[:1]

char_indices = { character : c_idx for c_idx, character in enumerate(character_set) }
index_chars = { c_idx : character for character, c_idx in char_indices.iteritems() }
