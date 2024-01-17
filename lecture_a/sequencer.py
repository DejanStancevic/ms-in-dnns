def fibonacci(n):
    if n <= 0:
        raise ValueError
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        i = 0
        seq = [0, 1]

        while i < n-2:
            seq.append(seq[-1] + seq[-2])
            i += 1

        return(seq)


def prime(n):
    if n <= 0:
        raise ValueError
    
    primes = [2]
    i = 3
    while len(primes) < n:
        is_prime = True

        for prime in primes:
            if i % prime == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(i)
        
        i += 1
    
    return(primes)

def square(n):
    if n <= 0:
        raise ValueError
    
    seq = [i*i for i in range(1, n+1)]
    return(seq)

def triangular(n):
    if n <= 0:
        raise ValueError
    
    seq = [(i*(i+1))//2 for i in range(1, n+1)]
    return(seq)

def factorial(n):
    if n <= 0:
        raise ValueError
    
    seq = [1]
    for i in range(2, n+1):
        seq.append(seq[-1]*i)
    
    return(seq)


def main(args):
    sequence = args.sequence
    n = args.length

    if sequence == 'fibonacci':
        return fibonacci(n)
    
    elif sequence == 'square':
        return square(n)

    elif sequence == 'triangular':
        return triangular(n)

    elif sequence == 'factorial':
        return factorial(n)

    elif sequence == 'prime':
        return prime(n)
    
    print('invalid choice', file = sys.stderr)
    



if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument( '--length', dest = 'length', type = int )
    parser.add_argument( '--sequence', dest = 'sequence', type = str )

    try:
       out =  main(parser.parse_args())
       if out == None:
           print('invalid choice')
       else:
           print(out)
    except:
        print( 'invalid choice', file = sys.stderr)





