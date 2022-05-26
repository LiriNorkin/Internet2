import sys


# Build inverted index.
def create_index():
    return None


# Return relavent docs based on given question.
def query():

   return None


if __name__ == '__main__':
    mood = sys.argv[1]
    if mood == "create_index":
        create_index(sys.argv[2])
    elif mood == "query":
        query()