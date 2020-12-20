import numpy as np

def find(cr, work, data, book):
    if cr == len(data[1]):
        work = np.array(work)
        return (work, np.sum((~(work[:-1] == work[1:]))))
    else:
        min = 10000
        result = []
        for worker in book[data[1][cr]]:
            work_ = work + [worker]
            worker_ = find(cr + 1, work_, data, book)
            if worker_[1] < min:
                min = worker_[1]
                result = worker_[0]
        return (result, min)

def eval(data):
    book_ = {}
    lst = []
    k = 0

    for i in data[0]:
        for j in i[1]:
            book_[j] = book_.get(j, []) + [i[0]]
            lst.append(j)

    for i in data[1]:
        if i in set(lst):
            k += 1

    if k == len(data[1]): return list(find(0, [], data, book_)[0])
    return []
