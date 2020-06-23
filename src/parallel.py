from multiprocessing import Pool

def parallel_imap(func, args, num_workers=10):
    results = []
    with Pool(num_workers) as pool:
        for result in pool.imap(func, args, chunksize=len(args)//num_workers):
            results.append(result)

    return results

def parallel_async(func, args, num_workers=10):
    def update_result(result):
        return result

    results = []
    async_promises = []
    with Pool(num_workers) as pool:
        for arg in args:
            r = pool.apply_async(func, arg, callback=update_result)
            async_promises.append(r)
        for r in async_promises:
            try:
                r.wait()
                results.append(r.get())
            except Exception as e:
                results.append(r.get())

    return results

def sequential(func, args):
    results = []
    for arg in args:
        result = func(arg)
        results.append(result)
    return results
