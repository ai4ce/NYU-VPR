import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a .txt file for testing')
    #parser.add_argument('--image_path', type=str, help='The folder contains all test images', required=True)
    #args = parser.parse_args()

    #files = os.listdir(args.image_path)
    files=open("test_image_paths.txt")
    f = open("test.txt","w")
    for line in files:
        f.write(line[:-1] + " 0.0 0.0\n")
    f.close()