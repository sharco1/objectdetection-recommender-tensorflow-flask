import os
path = '\\Users\\sharco\\Downloads\\Compressed\\pics\\pics'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, "image" + str(index) + ".jpg"))