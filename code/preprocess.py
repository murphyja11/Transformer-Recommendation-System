import numpy as np
from datetime import datetime

def preprocess(data_file, user_data_file):
	"""
	This function takes the filepaths to the tsv data files and preprocesses them

	First, it reads the tsv files to np arrays with entries as strings
	Then, it iterates through each file in the data file
	From the individual events, it creates a dictionary mapping artist IDs as strings to new integer ids
	It does the same for track ids
	It creates a user event dictionary that maps user ids (strings) to a list of single events.  These individual
	events have the schema [TimeStamp (python datetime object), Seconds Since Previous Listen (Integer), Artist ID
	(Integer), Track ID (Integer)]
	It then creates a user info dictionary mapping user id (string) to (Gender (Int), Age (Int), Country (Int)

	:param data_file: filepath to events data file
	:param user_data_file: filepath to user info data file
	:return: user events dictionary, user info dictionary, user id set, artist id dictionary, track id dictionary,
	country id dictionary
	"""
	# A dictionary where the keys are user ids and the values are np.arrays of timestamp, seconds since first song
	# artist id, artist name, track id, track name
	user_events = {}
	artist_ids = {} # maps artist ids to ints
	artist_id_acc = 0
	track_ids = {} # maps track ids to ints
	track_id_acc = 0
	num_songs = 0
	
	with open(data_file) as f:
		# Iterate through the lines in the TSV file
		for i in range(20):
			line = f.readline() # for line in reversed(f):
			event = np.array(line.replace('\n', '').rsplit('\t'))
			# array of user id, timestamp, artist id, artist name, track id, track name

			# Convert string of timestamp to Datetime object
			timestamp = datetime.strptime(event[1], '%Y-%m-%dT%H:%M:%SZ')  # Timestamp schema = %Y-%m-%dT%H:%M:%SZ

			# Update artist id dictionary
			if event[3] not in artist_ids.keys():
				artist_ids[event[3]] = artist_id_acc
				artist_id_acc += 1
			artist_id = artist_ids[event[3]]

			# Update track id dictionary
			if event[5] not in track_ids.keys():
				track_ids[event[5]] = track_id_acc
				track_id_acc += 1
			track_id = track_ids[event[5]]

			num_songs += 1

			# Add new event to the dictionary, appending to existing users
			if event[0] not in user_events.keys():  # check if listener exists in events dictionary
				# Value schema = timestamp, seconds since first timestamp, artist id, track id
				values = np.array([timestamp, 0, artist_id, track_id])
				user_events[event[0]] = [values]
			else:
				last_listen = user_events[event[0]][-1][0]  # timestamp of last listen
				secs_since_last_listen = (timestamp - last_listen).total_seconds()

				values = np.array([timestamp, secs_since_last_listen, artist_id, track_id])
				user_events[event[0]].append(values)


	user_info_dict = {} # A dictionary that mamps user id to user info
	with open(user_data_file) as f:
		f.readline() # read header line
		for i in range(20):
			line = f.readline() # for line in reversed(f):
			user = np.array(line.replace('\n', '').rsplit('\t'))
			# Map gender
			if user[1] == 'f':
				gender = 0
			elif user[1] == 'm':
				gender = 1
			else:  # Empty
				gender = 2

			# Map age
			try:
				age = int(user[2])
			except ValueError:
				age = 0

			country_dict = {}
			country_id = 0
			if user[3] not in country_dict.keys():
				country_dict[user[3]] = country_id
				country_id += country_id

			country = country_dict[user[3]]

			user_info_dict[user[0]] = np.array([gender, age, country])

	return user_events, user_info_dict, num_songs, artist_ids, track_ids, country_dict
