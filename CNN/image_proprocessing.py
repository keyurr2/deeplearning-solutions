import os
import shutil
import math

class ImagePreProcessor():
    def __init__(self, input_path = "train/", output_path = "test/", categories = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']) :
        self.input_path = input_path
        self.output_path = output_path
        self.categories = categories
    def generate_test_set(self, train = 0.25) :
        self.train = train
        for category in self.categories :
            input_dir = self.input_path + category
            output_dir = self.output_path + category
            files = [file for file in os.listdir(input_dir)]
            random_amount = math.floor(self.train * len(files))
            for i, file in enumerate(files, 1) :
                changed_file = category + "_" + str(i) + ".jpg"
                in_file = os.path.join(input_dir, changed_file)
                os.rename(os.path.join(input_dir, file), in_file)
                if i <= random_amount : 
                    shutil.copyfile(in_file, output_dir)

# ip = ImagePreProcessor()
# ip.generate_test_set()
