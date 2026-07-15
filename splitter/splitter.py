# splitter.py : Python code for Splitting Dataset into Training Data and Testing Data 
import random
import os
import csv
import argparse
import colorama
colorama.init(wrap=True)

parser = argparse.ArgumentParser(description="Split a list into training and testing sets based on a percentage.")
parser.add_argument('--percent', type=int, default=80, help='Percentage of data to be used for training (between 0 and 100).')
args = parser.parse_args()
percent = args.percent

if percent < 0 or percent > 100:
    print('\033[1m'+'\033[93m'+"The percentage must be between 0 and 100."+'\033[0m')
    quit()

def name_ext(nameext):
    name = nameext.rsplit('.', 1)[0]
    ext = nameext.rsplit('.', 1)[1]
    return name, ext

def divide_test_train_csv(fname):
    try:
        fhand = open(fname)
        print(f"Loading {extension[0]}.{extension[1]} ...")
    except:
        print(error_msg)
        quit()

    content = []
    for line in fhand:
        content.append(line.rstrip())

    header = f"{content[0]}"

    nrows = len(content)-1
    allrows = list(range(1,nrows+1))

    # ratio of training data and testing data
    ratio_train = percent/100
    ratio_test = 1-ratio_train

    nrow_test = int(ratio_test*nrows)
    nrow_train = nrows - nrow_test

    import time
    def rotating_symbol(duration=3, delay=0.1):
        start_time = time.time()
        end_time = start_time + duration
        symbols = ['-', '\\', '|', '/']
        while time.time() < end_time:
            for symbol in symbols:
                print(symbol, end='\r', flush=True)
                time.sleep(delay)
    print()
    rotating_symbol()

    list_test = random.sample(content[1:], k = nrow_test)
    list_train = [item for item in content[1:] if item not in list_test]

    list_test.insert(0,header)
    list_train.insert(0,header)

    def write_to_csv(filename, data):
        with open(filename, 'w') as file:
            for line in data:
                file.write(line + '\n')

    jname = fname.rsplit('.', 1)[0]
    write_to_csv(f"{jname}_testing.csv", list_test)
    write_to_csv(f"{jname}_training.csv", list_train)

    print(f"Number of Data (Rows) \t: {nrows}")
    print(f"Training Data Size \t: {nrow_train}")
    print(f"Testing Data Size \t: {nrow_test}")
    print("\nSuccesfully creating Training Data and Testing Data as")
    print('\t\033[1m'+'\033[92m'+f"{jname}_training.{extension[1]}"+'\033[0m', "and")
    print('\t\033[1m'+'\033[92m'+f"{jname}_testing.{extension[1]}"+'\033[0m')
    print("-fn\n")

def divide_test_train_x(fname, e):
    try:
        import pandas as pd
        df1 = pd.read_excel(fname)
        jname = fname.rsplit('.', 1)[0]
        tempFile = f"{jname}.csv"
        df1.to_csv(tempFile, index=False)
        divide_test_train_csv(tempFile)
        df2 = pd.read_csv(f"{jname}_testing.csv")
        df2.to_excel(f"{jname}_testing.{e}", index=False)
        df3 = pd.read_csv(f"{jname}_training.csv")
        df3.to_excel(f"{jname}_training.{e}", index=False)
        os.remove(f"{jname}.csv")
        os.remove(f"{jname}_testing.csv")
        os.remove(f"{jname}_training.csv")
    except:
        print(error_msg)
        quit()

banner = "                ---> Training    \n               |                 \n splitter -----|                 \n               |                 \n                ---> Testing     "
print('\n\033[1m'+'\033[43m'+banner+'\033[0m\n')
print("Python Program for Splitting Dataset into Training Data and Testing Data")
fname = input("Enter the csv or xlsx file : ") #"h2.csv"

error_msg = '\n\033[1m'+'\033[91m'+f"ERROR: The file {fname} is not available in the current directory."+'\033[0m\n-fn \n'
error_msg_2 = '\n\033[1m'+'\033[91m'+f"ERROR: The file {fname} is not suitable. Enter .xlsx or .csv files"+'\033[0m\n-fn \n'

try:
    extension = name_ext(fname)
except:
    print(error_msg)
    quit()

if extension[1] == "csv":
    divide_test_train_csv(fname)

elif extension[1] == "xlsx":
    divide_test_train_x(fname, extension[1])

else: print(error_msg_2)

