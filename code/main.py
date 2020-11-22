from preprocess import *
from model import *


def train(model, data, labels):
    """
    This function trains the model for one epoch

    :param model: The model
    :param data: The
    :param labels:
    :return:
    """

    for batch in range(0, len(data), model.batch_size):
        break

    pass


def test(model, data, labels):


    pass


def main():
    data_filepath = '../data/userid-timestamp-artid-artname-traid-traname.tsv'
    user_data_filepath = '../data/userid-profile.tsv'
    user_events, user_info_dict, artist_ids, track_ids, country_dict = preprocess(data_filepath, user_data_filepath)

    model = Model()
    epochs = 5
    for epoch in epochs:
        train(model, user_events)
        loss = test(model, user_events)

        print('Loss after epoch {} = {}'.format(epoch, loss))




if __name__ == '__main__':
    main()