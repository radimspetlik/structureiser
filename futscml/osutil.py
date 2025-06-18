import os

def dir_diff(dir1, dir2, verb=False):
    if not os.path.exists(dir1):
        print(f"{dir1}: path doesn't exist")
        return None
    if not os.path.exists(dir2):
        print(f"{dir2}: path doesn't exist")
        return None

    l1 = os.listdir(dir1)
    l2 = os.listdir(dir2)

    l1 = [a.split(".")[0] for a in l1]
    l2 = [a.split(".")[0] for a in l2]

    l3 = set(l1) - set(l2)
    l4 = set(l2) - set(l1)

    if verb:
        print("In dir1 but not in dir2:")
        for i in l3:
            print("\t" + i)
        print("In dir2 but not in dir1:")
        for i in l4:
            print("\t" + i)
    return l3, l4