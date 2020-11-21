from preprocess import *





def main():
    data_filepath = '../data/userid-timestamp-artid-artname-traid-traname.tsv'
    user_data_filepath = '../data/userid-profile.tsv'
    data, user_data = preprocess(data_filepath, user_data_filepath)

    epochs = 5
    model = Model()
    for epoch in epochs:
        train(model, )


if __name__ == '__main__':
    main()