import os

DATA_OUTPUT_DIRECTORY = 'out'
os.makedirs(DATA_OUTPUT_DIRECTORY, exist_ok=True)

IMAGE_FILE_EXTENSION = 'jpg'

MINION_2_DIRECTORY = 'minion_2_pics'
MINION_2_FRAMES = sorted([f'{MINION_2_DIRECTORY}/{file_name}' for file_name in os.listdir(MINION_2_DIRECTORY) if file_name.endswith(IMAGE_FILE_EXTENSION)])

MINION_3_DIRECTORY = 'minion_3_pics'
MINION_3_FRAMES = sorted([f'{MINION_3_DIRECTORY}/{file_name}' for file_name in os.listdir(MINION_3_DIRECTORY) if file_name.endswith(IMAGE_FILE_EXTENSION)])

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

print("const.py module loaded")
