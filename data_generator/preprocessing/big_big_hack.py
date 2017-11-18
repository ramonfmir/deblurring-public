path_truths = []

while (True):
    try:
        in_ = input().split(' ')
        path, ground_truth = in_[0], in_[1]
        if not ('*' in ground_truth or '#' in ground_truth):
            path_truths.append((path, ground_truth))
    except EOFError:
        print("All plates processed")
        break

print(path_truths)
