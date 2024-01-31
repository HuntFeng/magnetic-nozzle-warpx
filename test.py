import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-restart", "--restart", help="Set restart file")
args = parser.parse_args()
print(args.restart)
