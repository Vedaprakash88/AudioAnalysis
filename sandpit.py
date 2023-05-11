import os

root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Data\\genres_original\\"
for path, subdirs, files in os.walk(root):
    for name in files:
        print(os.path.join(path, name))



