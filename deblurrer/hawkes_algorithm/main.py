import sys

from char_enhancer import char_enhance
from char_detector import char_detect
from char_splitter import char_split

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    img = char_split(char_detect(char_enhance(img)))

    cv2.imshow('Regions', img)
    cv2.waitKey(0)
