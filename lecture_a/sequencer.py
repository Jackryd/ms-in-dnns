import argparse, math


def fib(length):
    if length <= 0:
        return []
    if length == 1:
        return [0]
    arr = [0]*length
    arr[1] = 1
    for i in range(2, length):
        arr[i] = arr[i-1]+arr[i-2]
    return arr

def isprime(val):
    if val < 2:
        return False
    for j in range(2, int(math.sqrt(val))+1):
        if val % j == 0:
            return False
    return True

def prime(length):
    arr = []
    c = 2
    while len(arr) < length:
        if isprime(c):
            arr.append(c)
        c += 1
    return arr

def square(length):
    arr = []
    for i in range(1, length+1):
        arr.append(i*i)
    return arr

def triang(length):
    arr = []
    for i in range(1, length+1):
        if len(arr) != 0:
            arr.append(arr[-1]+i)
        else:
            arr.append(1)
    return arr

def fact(length):
    arr = []
    for i in range(1, length+1):
        if len(arr) != 0:
            arr.append(arr[-1]*i)
        else:
            arr.append(1)
    return arr

def main(args):
    sequence, length  = args.sequence, args.length
    if sequence == "fibonacci":
        return fib(length)
    elif sequence == "prime":
        return prime(length)
    elif sequence == "square":
        return square(length)
    elif sequence == "triangular":
        return triang(length)
    elif sequence == "factorial":
        return fact(length)
    raise ValueError("invalid choice")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sequence", choices=["fibonacci", "prime", "square", "triangular", "factorial"], required=True)
    parser.add_argument( "-l", "--length", type=int, default=1)
    args = parser.parse_args()
    print(main(args))