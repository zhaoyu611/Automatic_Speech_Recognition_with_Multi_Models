import time

for _ in range(5):
    start_time = time.time()
    x = 1
    y = 2
    z = x+y
    end_time = time.time()

    print(end_time-start_time)
    