import permuter
import cv2
import os
import glob

if __name__ == '__main__':
    src = "../../data/40nicer/ch4.jpg"
    # files = glob.glob(os.path.abspath("../../data/40nicer/*g"))
    # count = 0
    # print(files)
    # for f in files:
    #     print(f)
    #     for i in range(7):
    #         char = permuter.get_n_char(f, i)
    #         print("Hacking...", count)
    #         dst = os.path.abspath("../../data/hackedIms/%i.jpg" % count)
    #         cv2.imwrite(dst, char)
    #         count += 1

    dst = os.path.abspath("../../data/hackedIms/T.jpg")
    char = permuter.get_n_char(os.path.abspath("../../data/40nicer/shandong2014.jpg"), 2)
    cv2.imwrite(dst, char)
