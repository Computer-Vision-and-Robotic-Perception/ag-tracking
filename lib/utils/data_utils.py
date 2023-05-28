import os

def clean_path(path):
    if os.path.exists(path):
        os.system('rm -rf "%s/*"' % path)
    else:         
        os.makedirs(path)

def get_mot_gt(file):
    file = open(file, 'r')
    lines = list(file.readlines())
    linesint = []
    for line in lines:
        fields = line[:-1].split(',')
        frame = int(fields[0])
        id = int(fields[1])
        x = int(fields[2])
        y = int(fields[3])
        w = int(fields[4])
        h = int(fields[5])
        linesint.append([frame, id, x, y, x + w, y + h])
    return linesint
