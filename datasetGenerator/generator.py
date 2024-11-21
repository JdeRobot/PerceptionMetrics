import argparse
import os


def configure_arguments():
    parser = argparse.ArgumentParser(description='Generate balanced datasets')
    parser.add_argument('--inputPath', type=str, help='input data path')
    return parser.parse_args()

def main(path):
    if not os.path.exists(path):
        raise "Path: " + str(path) + " does not exist"

    samples={}
    for root, dirs, files in os.walk(path, topdown=False):
        for d in dirs:
            currentPath=os.path.join(root,d)
            for f in os.listdir(currentPath):
                if f.endswith(".json"):
                    if currentPath not in samples:
                        samples[currentPath] = []
                    samples[currentPath].append(f)

    totalCount=0
    for sample in samples:
        print(sample, ":", len(samples[sample]))
        totalCount+=len(samples[sample])

    print("totalCount=", totalCount)




if __name__ == "__main__":
    args= configure_arguments()
    main(args.inputPath)






#arguments example:
# --inputPath=/mnt/large/pentalo/deep/datasets/synced
