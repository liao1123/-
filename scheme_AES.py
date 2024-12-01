from Crypto.Cipher import AES
from Crypto.Cipher._mode_ecb import EcbMode
from Crypto.Util.Padding import pad, unpad
from typing import Tuple, Dict, Any, Set
from tqdm import tqdm
from itertools import combinations
from numpy import frombuffer, bitwise_xor, uint64
from bitarray import bitarray
from math import comb
from collections import defaultdict
import numpy as np
import pyximport;

pyximport.install()
from splitter import split_bytes


def faster_xor(aa: bytes, bb: bytes):
    return bitwise_xor(frombuffer(aa, dtype=uint64), frombuffer(bb, dtype=uint64)).tobytes()


def encrypt(raw: bytes, key: str) -> bytes:
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
    return cipher.encrypt(raw)


def decrypt(enc, key, blocksize):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
    pt = cipher.decrypt(enc)
    try:
        pt = unpad(pt, blocksize)
    except:
        None
    return pt


def int2bytes(raw: int) -> bytes:
    """Convert integer to bytes"""
    return raw.to_bytes((raw.bit_length() + 7) // 8, byteorder='big')


def GenG(key: str, blocksize):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)

    def G(raw: int) -> bytes:
        """
        # >>> GenG('abcdefghijklmnop')(123)
        331992630117788453751424125151176572684
        """
        tep = raw.to_bytes((raw.bit_length() + 7) // 8, byteorder='big')
        if len(tep) == 0 or len(tep) % blocksize != 0:
            tep = pad(tep, blocksize)
        return cipher.encrypt(tep)

    return G


def G(raw: int, cipher: EcbMode, blocksize: int) -> bytes:
    """
    >>> G('abcdefghijklmnop', 123)
    331992630117788453751424125151176572684
    """
    tep = raw.to_bytes((raw.bit_length() + 7) // 8, byteorder='big')
    if len(tep) == 0 or len(tep) % blocksize != 0:
        tep = pad(tep, blocksize)
    return cipher.encrypt(tep)


def GenPF(key: str, blocksize):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)

    def PF(raw: bytes) -> bytes:
        if len(raw) % blocksize != 0:
            raw = pad(raw, blocksize)
        return cipher.encrypt(raw)

    return PF


# raw：原始数据 cipher: AES加密器, blocksize：分组密码AES
def PF(raw: bytes, cipher: EcbMode, blocksize: int):
    if len(raw) % blocksize != 0:
        raw = pad(raw, blocksize)
    return cipher.encrypt(raw)


def getCandidates(query: bytes, hashLen: int, R: int) -> list:
    """
    This function generates all possible candidates by flipping r bits of the input query string.

    Args:
    - query (str): The input query string.
    - r (int): The number of bits to flip.

    Returns:
    - List[str]: A list of all possible candidates by flipping r bits of the input query string.
    """
    query = bin(int.from_bytes(query, 'big'))[2:].zfill(hashLen)
    n = len(query)
    result = []
    for r in (range(R + 1)):
        for combination in combinations(list(range(n)), r):
            token = list(query)
            for i in combination:
                token[i] = '1' if token[i] == '0' else '0'
            result.append(int(''.join(token), 2).to_bytes(hashLen >> 3, byteorder='big'))
    return result


def partition_ordered(n, k, t):
    """
    This function returns all possible partitions of a positive integer n into k positive integers,
    where each integer is at most t.

    Args:
    - n (int): The positive integer to be partitioned.
    - k (int): The number of partitions.
    - t (int): The maximum value of each partition.

    Returns:
    - list: A list of all possible partitions of n into k positive integers, where each integer is at most t.
    """
    memo = {}

    def helper(n, k):
        if (n, k) in memo:
            return memo[(n, k)]
        if k == 1:
            if 0 <= n <= t:
                memo[(n, k)] = [[n]]
            else:
                memo[(n, k)] = []
            return memo[(n, k)]
        res = []
        for i in range(min(n, t) + 1):
            res.extend([[i] + sub for sub in helper(n - i, k - 1)])
        memo[(n, k)] = res
        return res

    return helper(n, k)


class SSECMRKey:
    """Class for SSECMRKey"""

    def __init__(self, K1: str, K2: str, K3: str):
        self.K1, self.K2, self.K3 = K1, K2, K3

    def __str__(self) -> str:
        return f"SSECMRKey({self.K1}, {self.K2}, {self.K3})"


class Label:
    """Class for Label"""
    __slots__ = ['blocksize', 'K1', 'K2', 'K3', 'K4']

    # 初始化四个密钥K
    def __init__(self, blocksize: int = 16):
        self.blocksize = blocksize

        self.K1 = 'a' * blocksize
        self.K2 = 'b' * blocksize
        self.K3 = 'c' * blocksize
        self.K4 = 'd' * blocksize

    def Gen(self) -> SSECMRKey:
        return SSECMRKey(self.K1, self.K2, self.K3)

    # 加密IMI索引结构
    def Enc(self, K, L):
        """Encrypt data"""
        K1, K2 = K.K1, K.K2
        K4 = self.K4

        cipher1 = AES.new(self.K1.encode('utf-8'), AES.MODE_ECB)  # F
        cipher2 = AES.new(self.K2.encode('utf-8'), AES.MODE_ECB)  # P
        cipher3 = AES.new(self.K3.encode('utf-8'), AES.MODE_ECB)  # Pi
        cipher4 = AES.new(self.K4.encode('utf-8'), AES.MODE_ECB)  # G

        # 明文采用论文方法加密
        return {PF(w, cipher2, self.blocksize): faster_xor(b''.join([G(i, cipher4, self.blocksize) for i in L[w]]),
                                                           len(L[w]) * PF(w, cipher1, self.blocksize)) for w in
                L.keys()}

    # # 对id解密
    # def Dec(self, K, en_id_list):
    #     K4 = self.K4
    #     cipher4 = AES.new(self.K4.encode('utf-8'), AES.MODE_ECB)  # G
    #
    #     id = []
    #     for en_id in en_id_list:
    #         de_id = cipher4.decrypt(en_id)
    #         de_id = de_id.rstrip(b'\r\x0e')
    #         id.append(int.from_bytes(de_id, byteorder='big'))
    #     return id
    # # 暂时用不到对原始数据加密
    # # 加密原始data但是 原始data要是二进制
    # def EncData(self, data):
    #     # Do not consider v for short
    #     cipher4 = AES.new(self.K4.encode('utf-8'), AES.MODE_ECB)
    #     return {G(key, cipher4, self.blocksize): key for key in tqdm(data, desc="Encrypting")}


class User:
    __slots__ = ['blocksize', 'hashLen', 'hashLenByte', 'ss']
    """
    A class representing User.

    Attributes:
    - blocksize (int): The block size used for encryption.
    - lenV (int): The length of the vector used for encryption.
    - hashLen (int): The length of the hash used for encryption.
    - hashLenByte (int): The length of the hash in bytes.
    - ss (int): The security parameter.
    - F (callable): A pseudorandom function generator.
    - P (callable): A pseudorandom permutation generator.

    Methods:
    - __init__(self, blocksize: int = 16, lenV: int = 0, hashLen: int = 16, K: SSECMRKey = None, ss = 0) -> None:
        Initializes a new instance of the MyScheme class.
    - Token(self, K: SSECMRKey, w: str) -> Tuple[int, int]:
        Generates a token for a given keyword.
    - Search(self, gamma: Dict[str, str], tao: Tuple[int, int]) -> Tuple[List[str], List[str]]:
        Searches for data in the encrypted index.
    - generateToken(self, query: bytes, r: int, K: SSECMRKey):
        Generates tokens for all possible query results with exactly r flipped bits.
    """

    def __init__(self, blocksize: int, hashLen: int, K: SSECMRKey, ss: int):
        """
        Initializes a new instance of the SEIndex class.

        Args:
        - blocksize (int): The block size used for encryption.
        - lenV (int): The length of the vector used for encryption.
        - hashLen (int): The length of the hash used for encryption.
        - K (SSECMRKey): The SSECMRKey object used for encryption.
        - ss (int): The security parameter.
        """
        self.blocksize = blocksize
        self.hashLen = hashLen
        self.hashLenByte = hashLen >> 3
        self.ss = ss

    # 输入密钥和关键字w来生成一个搜索令牌
    def Token(self, K: SSECMRKey, w: str) -> Tuple[int, int]:
        """
        Generates a token for a given keyword.

        Args:
        - K (SSECMRKey): The SSECMRKey object used for encryption.
        - w (str): The keyword to generate a token for.

        Returns:
        - A tuple of two integers representing the generated token.
        """
        cipher1 = AES.new(K.K1.encode('utf-8'), AES.MODE_ECB)  # F
        cipher2 = AES.new(K.K2.encode('utf-8'), AES.MODE_ECB)  # P
        return (PF(w, cipher1, self.blocksize), PF(w, cipher2, self.blocksize))

    # 计算翻转多个位之后的可能查询，之后对其进行加密生成搜索令牌
    def generateToken(self, query: bytes, r: int, K: SSECMRKey):
        '''
        Filp exactly r bits in query, get all the possile results, and return token generated by these results.
        The size of result is C_len(query)^r.
        Args:
        - query (bytes): The query to generate tokens for.
        - r (int): The number of bits to flip in the query.
        - K (SSECMRKey): The SSECMRKey object used for encryption.
        '''
        baquery = bitarray()
        baquery.frombytes(query)
        indices = list(range(len(baquery)))
        result = [None] * comb(len(baquery), r)

        cipher1 = AES.new(K.K1.encode('utf-8'), AES.MODE_ECB)  # F
        cipher2 = AES.new(K.K2.encode('utf-8'), AES.MODE_ECB)  # P

        for idx, combination in enumerate(combinations(indices, r)):
            res = bitarray()
            res.frombytes(query)
            for i in combination:
                res[i] = 1 if baquery[i] == 0 else 0
            result[idx] = (PF(res.tobytes(), cipher1, self.blocksize), PF(res.tobytes(), cipher2, self.blocksize))
        return result

    # 多表令牌的生成
    def tokenGen(self, query, R, K):
        """
        Return the list of all possible tokens for all ss tables.
        d1 Res[i]: list refers to the i-th table's token.
        d2 Res[i][j]: list refers to the i-th table's token that have hamming distance of j with query.
        d3 Res[i][j][k]: turple refers to the k-th token of the i-th table's token that have hamming distance of j with query.
        """
        Res = [{} for _ in range(self.ss)]
        cipher1 = AES.new(K.K1.encode('utf-8'), AES.MODE_ECB)  # F
        cipher2 = AES.new(K.K2.encode('utf-8'), AES.MODE_ECB)  # P
        indices = list(range(8 * self.hashLenByte // self.ss))

        for r in range(R + 1):
            for sIdx in range(self.ss):
                left, right = (self.hashLenByte // self.ss) * sIdx, (self.hashLenByte // self.ss) * (sIdx + 1)
                baquery = bitarray()
                baquery.frombytes(query[left: right])
                result = [None] * comb(len(baquery), r)
                for idx, combination in enumerate(combinations(indices, r)):
                    res = baquery.copy()
                    for k in combination:
                        res[k] = not res[k]
                    result[idx] = (
                        (PF(res.tobytes(), cipher1, self.blocksize), PF(res.tobytes(), cipher2, self.blocksize)))
                Res[sIdx][r] = result

        return Res

    # 在加密索引gamma数据库里面通过tao索引来查找对应索引
    def Search(self, gamma: Dict, tao: Tuple) -> Set[Any]:
        """
        Searches for data in the encrypted index.

        Args:
        - gamma (Dict[str, str]): The encrypted index. One of the II in enIMI.
        - tao (Tuple[int, int]): The search token.

        Returns:
        - A tuple of two lists representing the search results.
        """
        t = gamma.get(tao[1])
        if t is None:
            return set()
        return set(split_bytes(faster_xor(t, tao[0] * (len(t) // self.blocksize))))

    def hammingSearch(self, enIMI, token, r, partitions_dict):
        """
            Searches for data in the encrypted index under hamming distance.

            Args:
            - enIMI (List[Dict[str, str]]): The encrypted index.
            - token (List[Tuple[int, int]]): The search token that act as Res.
            - r (int): The number of bits to flip in the query.

            Returns:
            - A set representing the search results.
            """
        Res = [defaultdict(set) for _ in range(self.ss)]
        maxSize = self.hashLen // self.ss
        finalresult = [None] * (r + 1)
        min_val = min(r + 1, maxSize)
        search = self.Search

        # Prepare candidates for combination.
        for idx in range(self.ss):
            for ri in range(min_val):
                for tao in token[idx][ri]:
                    Res[idx][ri].update(search(enIMI[idx], tao))

        # Intersection for different combination.
        for i, ri in enumerate(range(r + 1)):
            tep = set()
            for R in partitions_dict[(ri, self.ss)]:
                tep.update(set.intersection(*[Res[idx][rj] for idx, rj in enumerate(R)]))
            finalresult[i] = tep
        return finalresult
