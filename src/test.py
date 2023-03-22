import time
import argparse
import sys


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, help="device")
	parser.add_argument("--outfile", type=str, help="Path to outfile")
	args = parser.parse_args()

	sys.stdout = open(args.outfile, 'wt')
	print(args.device)
	time.sleep(60)
	print(args.device)