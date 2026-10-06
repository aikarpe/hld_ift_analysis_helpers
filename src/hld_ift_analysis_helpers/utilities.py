import shutil

def free_disk_space_check(threshold: float = 20.0):

    total,used,free = shutil.disk_usage('/')

    print(f'Total: {total / 2**30} GiB')
    print(f' Used: {used  / 2**30} GiB')
    print(f' Free: {free  / 2**30} GiB')

    if free / 2**30 < threshold: 
        print(f"Free some space on disk!!! At least {threshold} GiB needed")
        exit()
    


