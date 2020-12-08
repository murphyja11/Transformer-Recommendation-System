from preprocess import *
from model import *
import random


def train(model, user_events, user_info, songs):
    """
    This function trains the model for one epoch

    :param model: the model
    :param user_events: a dictionary of user id to event info
    :param user_info: a dictionary of user id to user info
    :return:
    """
    user_ids = list(user_events.keys())
    random.shuffle(user_ids)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    for batch in range(0, len(user_ids), model.batch_size):
        event_list = []
        event_labels = []
        info_list = []
        for user in user_ids[batch:model.batch_size]:
            # get the inputs and labels for all user events
            events = user_events[user]
            event_list.append(events[:-1])
            event_labels.append(events[1:])

            info = user_info[user]
            info_list.append(info)

        #for ting in event_list[0]:
            #print(ting)
        event_list = tf.convert_to_tensor(np.asarray(event_list).astype('float32'))
        event_labels = tf.convert_to_tensor(np.asarray(event_labels).astype('float32'))
        info_list = tf.convert_to_tensor(np.asarray(info_list).astype('float32'))

        with tf.GradientTape() as tape:
            probs = model.call(event_list, info_list)
            loss = model.loss(probs, event_labels, None)

        gradients = tf.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if batch % 1000 == 0:
            print('Batch {} loss {}'.format(batch // model.batch_size, loss))


def test(model, user_events, user_info):
    user_ids = list(user_events.keys())
    loss = []

    for batch in range(0, len(user_ids), model.batch_size):
        event_list = []
        event_labels = []
        info_list = []
        for user in user_ids[batch:model.batch_size]:
            # get the inputs and labels for all user events
            events = user_events[user]
            event_list.append(events[:-1])
            event_labels.append(events[1:])

            info = user_info[user]
            info_list.append(info)

        event_list = tf.convert_to_tensor(np.asarray(event_list).astype('float32'))
        event_labels = tf.convert_to_tensor(np.asarray(event_labels).astype('float32'))
        info_list = tf.convert_to_tensor(np.asarray(info_list).astype('float32'))

        probs = model.call(event_list, info_list)
        loss.append(model.loss(probs, event_labels, None))

    return np.mean(loss)


def main():
    data_filepath = '../data/userid-timestamp-artid-artname-traid-traname.tsv'
    user_data_filepath = '../data/userid-profile.tsv'
    user_events, user_info_dict, num_songs, artist_ids, track_ids, country_dict = preprocess(data_filepath, user_data_filepath)
    user_ids = list(user_events.keys())
    random.shuffle(user_ids)
    user_ids = np.array(user_ids)

    # Train/Test split
    train_user_ids = user_ids[:int(len(user_ids) * .85)]
    test_user_ids = user_ids[int(len(user_ids) * .85):]

    train_user_events = {}
    train_user_info = {}
    for user_id in train_user_ids:
        train_user_events[user_id] = user_events[user_id]
        train_user_info[user_id] = user_info_dict[user_id]

    test_user_events = {}
    test_user_info = {}
    for user_id in test_user_ids:
        test_user_events[user_id] = user_events[user_id]
        test_user_info[user_id] = user_info_dict[user_id]

    model = Model(num_songs)
    epochs = 5
    for epoch in range(epochs):
        train(model, train_user_events, train_user_info, track_ids)
        loss = test(model, test_user_events, test_user_info)

        print('Loss after epoch {} = {}'.format(epoch, loss))


if __name__ == '__main__':
    main()
