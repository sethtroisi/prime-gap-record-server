# Copyright 2026 Seth Troisi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Source from unlicensed https://github.com/wacchoz/APR_CL
With improvements from Seth Troisi
See "Primality Testing and Jacobi Sums By H. Cohen and H.W. Lenstra, Jn."
"""

import copy
import math
import sys
import time

import gmpy2
import primesieve


# All factors of possible t
T_FACTORS = [2, 3, 5, 7, 11, 13, 17, 19]

# Count of prime factor q in t
def v(q, t):
    assert t > 0 and q > 0
    ans = 0
    while True:
        t, m = divmod(t, q)
        if m != 0:
            return ans
        ans += 1


def prime_factorize(n):
    ret = []
    for p in T_FACTORS:
        k = v(p, n)
        if k > 0:
            n, m = divmod(n, p ** k)
            assert m == 0
            ret.append((p, k))

    assert n == 1
    return ret


# calculate e(t)
def e(t):
    assert t % 2 == 0, "Odd case not handled"

    e_t = 1
    Q = []
    # Could be precomputed but meh
    for q in primesieve.primes(t+1):
        if t % (q-1) == 0:
            e_t *= q ** (v(q,t) + 1)
            #print(t, q, v(q, t))
            Q.append(q)
    return 2*e_t, Q

assert e(2)[0] == 24
assert e(12)[0] == 65520
assert e(60)[0] == 6814407600, e(60)

# Jacobi sum
class JacobiSum(object):
    def __init__(self, p, k, q):
        self.p = p
        self.k = k
        self.q = q
        self.m = (p-1)*p**(k-1)
        self.pk = p**k
        self.coef = [0]*self.m

    # 1
    def one(self):
        self.coef[0] = 1
        for i in range(1,self.m):
            self.coef[i] = 0
        return self


    # product of JacobiSum
    # jac : JacobiSum
    def mul(self, jac):
        m = self.m
        pk = self.pk
        j_ret=JacobiSum(self.p, self.k, self.q)
        for i in range(m):
            for j in range(m):
                if (i+j)% pk < m:
                    j_ret.coef[(i+j)% pk] += self.coef[i] * jac.coef[j]
                else:
                    step = self.p ** (self.k-1)
                    r = (i+j) % pk - step
                    while r>=0:
                        j_ret.coef[r] -= self.coef[i] * jac.coef[j]
                        r -= step

        return j_ret


    def __mul__(self, right):
        if type(right) is int:
            # product with integer
            j_ret=JacobiSum(self.p, self.k, self.q)
            for i in range(self.m):
                j_ret.coef[i] = self.coef[i] * right
            return j_ret
        else:
            # product with JacobiSum
            return self.mul(right)


    # power of JacobiSum（x-th power mod n）
    def modpow(self, x, n):
        j_ret = JacobiSum(self.p, self.k, self.q)
        j_ret.coef[0] = 1
        j_a = copy.deepcopy(self)
        while x > 0:
            if x % 2 == 1:
                j_ret = (j_ret * j_a).mod(n)
            j_a = j_a * j_a
            j_a.mod(n)
            x //= 2
        return j_ret


    # applying "mod n" to coefficient of self
    def mod(self, n):
        for i in range(self.m):
            self.coef[i] %= n
        return self


    # operate sigma_x
    # verification for sigma_inv
    def sigma(self, x):
        m = self.m
        pk = self.pk
        j_ret=JacobiSum(self.p, self.k, self.q)
        for i in range(m):
            if (i*x) % pk < m:
                j_ret.coef[(i*x) % pk] += self.coef[i]
            else:
                r = (i*x) % pk - self.p ** (self.k-1)
                while r>=0:
                    j_ret.coef[r] -= self.coef[i]
                    r-= self.p ** (self.k-1)
        return j_ret


    # operate sigma_x^(-1)
    def sigma_inv(self, x):
        m = self.m
        pk = self.pk
        j_ret=JacobiSum(self.p, self.k, self.q)
        for i in range(pk):
            if i<m:
                if (i*x)%pk < m:
                    j_ret.coef[i] += self.coef[(i*x)%pk]
            else:
                r = i - self.p ** (self.k-1)
                while r>=0:
                    if (i*x)%pk < m:
                        j_ret.coef[r] -= self.coef[(i*x)%pk]
                    r-= self.p ** (self.k-1)

        return j_ret


    # Is self p^k-th root of unity (mod N)
    # if so, return h where self is zeta^h
    def is_root_of_unity(self, N):
        m = self.m
        p = self.p
        k = self.k

        # case of zeta^h (h<m)
        one = 0
        for i in range(m):
            if self.coef[i]==1:
                one += 1
                h = i
            elif self.coef[i] == 0:
                continue
            elif (self.coef[i] - (-1)) %N != 0:
                return False, None
        if one == 1:
            return True, h

        # case of zeta^h (h>=m)
        for i in range(m):
            if self.coef[i]!=0:
                break
        r = i % (p**(k-1))
        for i in range(m):
            if i % (p**(k-1)) == r:
                if (self.coef[i] - (-1))%N != 0:
                    return False, None
            else:
                if self.coef[i] !=0:
                    return False, None

        return True, (p-1)*p**(k-1)+ r


# find primitive root
def smallest_primitive_root(q):
    for r in range(2, q):
        s = set()
        m = 1
        for i in range(1, q):
            m = (m*r) % q
            s.add(m)
        if len(s) == q-1:
            return r
    return None   # error


# calculate f_q(x)
def calc_f(q):
    g = smallest_primitive_root(q)
    m = {}
    for x in range(1,q-1):
        m[pow(g,x,q)] = x
    f = {}
    for x in range(1,q-1):
        f[x] = m[ (1-pow(g,x,q))%q ]

    return f


# sum zeta^(a*x+b*f(x))
def calc_J_ab(p, k, q, a, b):
    j_ret = JacobiSum(p,k,q)
    f = calc_f(q)
    for x in range(1,q-1):
        pk = p**k
        if (a*x+b*f[x]) % pk < j_ret.m:
            j_ret.coef[(a*x+b*f[x]) % pk] += 1
        else:
            r = (a*x+b*f[x]) % pk - p**(k-1)
            while r>=0:
                j_ret.coef[r] -= 1
                r-= p**(k-1)
    return j_ret


# calculate J(p,q)（p>=3 or p,q=2,2）
def calc_J(p, k, q):
    return calc_J_ab(p, k, q, 1, 1)


# calculate J_3(q)（p=2 and k>=3）
def calc_J3(p, k, q):
    j2q = calc_J(p, k, q)
    j21 = calc_J_ab(p, k, q, 2, 1)
    j_ret = j2q * j21
    return j_ret


# calculate J_2(q)（p=2 and k>=3）
def calc_J2(p, k, q):
    j31 = calc_J_ab(2, 3, q, 3, 1)
    j_conv = JacobiSum(p, k, q)
    for i in range(j31.m):
        j_conv.coef[i*(p**k)//8] = j31.coef[i]
    j_ret = j_conv * j_conv
    return j_ret


# in case of p>=3
def APRtest_step4a(p, k, q, N):
    J = calc_J(p, k, q)
    # initialize s1=1
    s1 = JacobiSum(p,k,q).one()
    # J^Theta
    for x in range(p**k):
        if x % p == 0:
            continue
        t = J.sigma_inv(x)
        t = t.modpow(x, N)
        s1 = s1 * t
        s1.mod(N)

    # r = N mod p^k
    r = N % (p**k)

    # s2 = s1 ^ (N/p^k)
    s2 = s1.modpow(N//(p**k), N)

    # J^alpha
    J_alpha = JacobiSum(p,k,q).one()
    for x in range(p**k):
        if x % p == 0:
            continue
        t = J.sigma_inv(x)
        t = t.modpow((r*x)//(p**k), N)
        J_alpha = J_alpha * t
        J_alpha.mod(N)

    # S = s2 * J_alpha
    S = (s2 * J_alpha).mod(N)

    # Is S root of unity
    exist, h = S.is_root_of_unity(N)

    if not exist:
        # composite!
        return False, None
    else:
        # possible prime
        if h%p!=0:
            l_p = 1
        else:
            l_p = 0
        return True, l_p


# in case of p=2 and k>=3
def APRtest_step4b(p, k, q, N):
    J = calc_J3(p, k, q)
    # initialize s1=1
    s1 = JacobiSum(p,k,q).one()
    # J3^Theta
    for x in range(p**k):
        if x % 8 not in [1,3]:
            continue
        t = J.sigma_inv(x)
        t = t.modpow(x, N)
        s1 = s1 * t
        s1.mod(N)

    # r = N mod p^k
    r = N % (p**k)

    # s2 = s1 ^ (N/p^k)
    s2 = s1.modpow(N//(p**k), N)

    # J3^alpha
    J_alpha = JacobiSum(p,k,q).one()
    for x in range(p**k):
        if x % 8 not in [1,3]:
            continue
        t = J.sigma_inv(x)
        t = t.modpow((r*x)//(p**k), N)
        J_alpha = J_alpha * t
        J_alpha.mod(N)

    # S = s2 * J_alpha * J2^delta
    if N%8 in [1,3]:
        S = (s2 * J_alpha ).mod(N)
    else:
        J2_delta = calc_J2(p,k,q)
        S = (s2 * J_alpha * J2_delta).mod(N)

    # Is S root of unity
    exist, h = S.is_root_of_unity(N)

    if not exist:
        # composite
        return False, None
    else:
        # possible prime
        if h%p!=0 and (pow(q,(N-1)//2,N) + 1)%N==0:
            l_p = 1
        else:
            l_p = 0
        return True, l_p


# in case of p=2 and k=2
def APRtest_step4c(p, k, q, N):
    J2q = calc_J(p, k, q)

    # s1 = J(2,q)^2 * q (mod N)
    s1 = (J2q * J2q * q).mod(N)

    # s2 = s1 ^ (N/4)
    s2 = s1.modpow(N//4, N)

    if N%4 == 1:
        S = s2
    elif N%4 == 3:
        S = (s2 * J2q * J2q).mod(N)
    else:
        print("Error")

    # Is S root of unity
    exist, h = S.is_root_of_unity(N)

    if not exist:
        # composite
        return False, None
    else:
        # possible prime
        if h%p!=0 and (pow(q,(N-1)//2,N) + 1)%N==0:
            l_p = 1
        else:
            l_p = 0
        return True, l_p


# in case of p=2 and k=1
def APRtest_step4d(p, k, q, N):
    S2q = pow(-q, (N-1)//2, N)
    if (S2q-1)%N != 0 and (S2q+1)%N != 0:
        # composite
        return False, None
    else:
        # possible prime
        if (S2q + 1)%N == 0 and (N-1)%4==0:
            l_p=1
        else:
            l_p=0
        return True, l_p


# Step 4
def APRtest_step4(p, k, q, N, verbose):

    if p>=3:
        if verbose:
            print("Step 4a. (p^k, q = {0}^{1}, {2})".format(p,k,q))
        result, l_p = APRtest_step4a(p, k, q, N)
    elif p==2 and k>=3:
        if verbose:
            print("Step 4b. (p^k, q = {0}^{1}, {2})".format(p,k,q))
        result, l_p = APRtest_step4b(p, k, q, N)
    elif p==2 and k==2:
        if verbose:
            print("Step 4c. (p^k, q = {0}^{1}, {2})".format(p,k,q))
        result, l_p = APRtest_step4c(p, k, q, N)
    elif p==2 and k==1:
        if verbose:
            print("Step 4d. (p^k, q = {0}^{1}, {2})".format(p,k,q))
        result, l_p = APRtest_step4d(p, k, q, N)
    else:
        assert False

    return result, l_p


def select_t(N):
    t_candidates = [
        2,
        12,
        60,
        180,
        840,
        1260,
        1680,
        2520,
        5040,
        15120,
        55440,
        110880,
        720720,
        1441440,
        4324320,
        24504480,
        73513440,
        367567200,
        1396755360,
        6983776800,
    ]

    # Select t
    for t in t_candidates:
        e_t, qlist = e(t)
        if e_t*e_t > N:
            return t, e_t, qlist

    assert False


def step1and2(N, verbose):
    t, e_t, Q = select_t(N)
    if verbose:
        print(f"e({t}) = {e_t} | {Q}")

    # Step 1
    if verbose:
        print("=== Step 1 ===")
    g = math.gcd(t * e_t, N)
    if g > 1:
        return False

    # Step 2
    if verbose:
        print("=== Step 2 ===")
    l = {}
    fac_t = prime_factorize(t)
    for p, k in fac_t:
        if p>=3 and pow(N, p-1, p*p)!=1:
            l[p] = 1
        else:
            l[p] = 0
    if verbose:
        print(f"l_p = {l}")
    return t, e_t, Q, l


def APRtest(N, verbose=False):
    if verbose:
        print(f"{N=}")

    if N <= 3:
        return N == 2 or N == 3

    result = step1and2(N, verbose)
    if result is False:
        return False
    t, e_t, Q, l = result

    # Step 3 & Step 4
    if verbose:
        print("=== Step 3&4 ===")
    for q in Q:
        if q == 2:
            continue
        fac = prime_factorize(q-1)
        for p,k in fac:

            # Step 4
            result, l_p = APRtest_step4(p, k, q, N, verbose)

            if not result:
                return False
            elif l_p==1:
                l[p] = 1

    # Step 5
    if verbose:
        print("=== Step 5 ===")
        print("l_p=", l)
    for p, value in l.items():
        if value==0:
            # try other pair of (p,q)
            if verbose:
                print("Try other (p,q). p={}".format(p))
            count = 0
            i = 1
            found = False
            # try maximum 30 times
            while count < 30:
                q = p*i+1
                if N%q != 0 and gmpy2.is_prime(q) and (q not in Q):
                    count += 1

                    k = v(p, q-1)
                    # Step 4
                    result, l_p = APRtest_step4(p, k, q, N, verbose)

                    if not result:
                        return False
                    elif l_p == 1:
                        found = True
                        break
                i += 1

            if not found:
                if verbose:
                    print("error in Step 5")
                return False

    # Step 6
    if verbose:
        print("=== Step 6 ===")
    r = 1
    for _ in range(t-1):
        r = (r*N) % e_t
        if r != 1 and r != N and N % r == 0:
            return False

    return True


if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 1208925819614629174706189
        N = 2**521-1  # 157 digit, 4.1
        N = 2**1279-1 # 386 digit, 122 seconds

    status = APRtest(N, verbose=True)

    end_time = time.time()
    print(end_time - start_time, "sec")

    exit(status)
