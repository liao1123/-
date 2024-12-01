cpdef split_bytes(byte_string):
    cdef bytes b = byte_string
    cdef int n = len(b)
    return [b[i:i+16] for i in range(0, n, 16)]
