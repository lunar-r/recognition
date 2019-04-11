def show_data(offset, idx):
    count = -1
    for i in range(100):
        count = count + 1
        if count < offset:
            continue
        if count >= offset + idx:
            break
        print(count)


show_data(0, 5)