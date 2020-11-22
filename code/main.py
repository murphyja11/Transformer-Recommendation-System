from preprocess import *


def train(model, data, labels):
    """
    This function trains the model for one epoch

    :param model: The model
    :param data: The
    :param labels:
    :return:
    """



    pass


def test(model, data, labels):


    pass

def main():
    data_filepath = '../data/userid-timestamp-artid-artname-traid-traname.tsv'
    user_data_filepath = '../data/userid-profile.tsv'
    user_events, user_info_dict, artist_ids, track_ids, country_dict = preprocess(data_filepath, user_data_filepath)

    epochs = 5
    #for epoch in epochs:

    print('Yo')


if __name__ == '__main__':
    main()