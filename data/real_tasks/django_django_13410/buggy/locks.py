            ('hEvent', HANDLE)]

    LPOVERLAPPED = POINTER(OVERLAPPED)

    # --- Define function prototypes for extra safety ---
    LockFileEx = windll.kernel32.LockFileEx
    LockFileEx.restype = BOOL
    LockFileEx.argtypes = [HANDLE, DWORD, DWORD, DWORD, DWORD, LPOVERLAPPED]
    UnlockFileEx = windll.kernel32.UnlockFileEx
    UnlockFileEx.restype = BOOL
    UnlockFileEx.argtypes = [HANDLE, DWORD, DWORD, DWORD, LPOVERLAPPED]

    def lock(f, flags):
        hfile = msvcrt.get_osfhandle(_fd(f))
        overlapped = OVERLAPPED()
        ret = LockFileEx(hfile, flags, 0, 0, 0xFFFF0000, byref(overlapped))
        return bool(ret)

    def unlock(f):
        hfile = msvcrt.get_osfhandle(_fd(f))
        overlapped = OVERLAPPED()
        ret = UnlockFileEx(hfile, 0, 0, 0xFFFF0000, byref(overlapped))
        return bool(ret)
else:
    try:
        import fcntl
        LOCK_SH = fcntl.LOCK_SH  # shared lock
        LOCK_NB = fcntl.LOCK_NB  # non-blocking
        LOCK_EX = fcntl.LOCK_EX
    except (ImportError, AttributeError):
        # File locking is not supported.
        LOCK_EX = LOCK_SH = LOCK_NB = 0

        # Dummy functions that don't do anything.
        def lock(f, flags):
            # File is not locked
            return False

        def unlock(f):
            # File is unlocked
            return True
    else:
        def lock(f, flags):
            ret = fcntl.flock(_fd(f), flags)
            return ret == 0

        def unlock(f):
            ret = fcntl.flock(_fd(f), fcntl.LOCK_UN)
            return ret == 0
