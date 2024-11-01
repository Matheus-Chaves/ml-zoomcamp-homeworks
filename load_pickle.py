import pickle
from os.path import join

FOLDER = 'data'

# for Q6 (Docker container)
with open('model2.bin', 'rb') as f:
    model = pickle.load(f)

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

# for the other questions
# with open(join(FOLDER, 'model1.bin'), 'rb') as f:
#     model = pickle.load(f)

# with open(join(FOLDER, 'dv.bin'), 'rb') as f:
#     dv = pickle.load(f)