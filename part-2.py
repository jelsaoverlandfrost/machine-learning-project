# TODO: Write a parser for the input files
# For better referencing:
SENTIMENT_TAGS = {}
CHUNK_TAGS = {}


# This function is for reading the input files
def parse_input_files(file_name):
    result = []
    with open(file_name, 'r') as file:
        while True:
            data_read = file.readline()
            if data_read == '':
                break
            print(data_read.replace('\n', ''))


parse_input_files("data\\EN\\train")
