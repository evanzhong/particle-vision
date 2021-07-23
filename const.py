import os

DATA_OUTPUT_DIRECTORY = 'out'
os.makedirs(DATA_OUTPUT_DIRECTORY, exist_ok=True)

IMAGE_FILE_EXTENSION = 'jpg'

MINION_2_DIRECTORY = 'minion_2_pics'
MINION_2_FRAMES = sorted([f'{MINION_2_DIRECTORY}/{file_name}' for file_name in os.listdir(MINION_2_DIRECTORY) if file_name.endswith(IMAGE_FILE_EXTENSION)])

MINION_3_DIRECTORY = 'minion_3_pics'
MINION_3_FRAMES = sorted([f'{MINION_3_DIRECTORY}/{file_name}' for file_name in os.listdir(MINION_3_DIRECTORY) if file_name.endswith(IMAGE_FILE_EXTENSION)])

print("const.py module loaded")
