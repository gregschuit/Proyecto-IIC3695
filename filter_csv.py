from os.path import isfile
from csv import DictReader
from pprint import pprint

with open("train0.csv") as file:
    data = DictReader(file)
    # pprint(list(filtered))
    filtered = filter(lambda x: isfile('images/train/' + x['id'].zfill(16) + '.jpg'), data)

    with open("train.csv", "w") as newfile:
        newfile.write("filename,url,landmark_id\n")
        try:
            line = next(filtered)
        except StopIteration:
            line = False
        counter = 0
        while line:
            print(counter, end="\r")
            newfile.write(f"{line['id']},{line['url']},{line['landmark_id']}\n")
            try:
                line = next(filtered)
            except StopIteration:
                line = False
            counter += 1