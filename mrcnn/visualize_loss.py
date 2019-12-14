if __name__=='__main__':
    import json
    import matplotlib.pyplot as plt
    with open("logs/50epochs.json") as json_file:
        data = json.load(json_file)

    epochs = []
    loss = []
    for d in data:
        epochs.append(d[1])
        loss.append(d[2])

    print(loss)
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.plot(epochs, loss)
    ax.set_title("Average loss vs epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')