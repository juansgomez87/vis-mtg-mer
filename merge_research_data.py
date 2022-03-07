#!/usr/bin/env python3
"""
Merge playlist information with annotations


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import json
import pdb
import pandas as pd
from tqdm import tqdm


from collections import Counter

def load_json(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = json.loads(data)
    return data


def aro_val_to_quads(aro, val):
	aro, val = int(aro), int(val)
	if aro == 1 and val == 1:
		quad = 1
	elif aro == 1 and val == -1:
		quad = 2
	elif aro == -1 and val == -1:
		quad = 3
	elif aro == -1 and val == 1:
		quad = 4
	return quad


def main(anno, df):
	tags = anno.moodValue.unique()
	songs = df.cdr_track_num.unique().tolist()
	for s in tqdm(songs):
		cnt_quad = Counter({_: 0 for _ in range(1, 5)})
		cnt_mood = Counter({_: 0 for _ in tags})
		this_song = anno[anno.externalID == s]
		cnt_quad.update(this_song.quadrant)
		cnt_mood.update(this_song.moodValue)
		num_users = this_song.shape[0]

		df.loc[df.cdr_track_num == s, 'num_users'] = num_users
		df.loc[df.cdr_track_num == s, cnt_quad.keys()] = cnt_quad.values()
		df.loc[df.cdr_track_num == s, cnt_mood.keys()] = cnt_mood.values()

		txt_free = ' '.join([_ for _ in this_song.freeMood.tolist() if _])
		df.loc[df.cdr_track_num == s, 'txt_free'] = txt_free
		txt_arousal = ' '.join([_ for _ in this_song.arousalComment.tolist() if _]).replace(',', ' ')
		txt_valence = ' '.join([_ for _ in this_song.valenceComment.tolist() if _]).replace(',', ' ')
		txt_quad = txt_arousal + txt_valence
		df.loc[df.cdr_track_num == s, 'txt_quad'] = txt_quad
		txt_mood = ' '.join([_ for _ in this_song.moodComment.tolist() if _]).replace(',', ' ')
		df.loc[df.cdr_track_num == s, 'txt_mood'] = txt_mood

		df.loc[df.cdr_track_num == s, 'pref'] = Counter(this_song.favSong)['1']/(num_users + 0.01)
		df.loc[df.cdr_track_num == s, 'fam'] = Counter(this_song.knownSong)['1']/(num_users + 0.01)

	df.to_csv('./data/summary_anno.csv', sep='\t')
	pdb.set_trace()


if __name__ == "__main__":
	# fn = './data/data_24_11_2021.json'
	fn = './data/data_07_03_2022.json'
	csv = './data/summary.csv'
	tags = ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace', 'tenderness', 'transcendence']
	data = load_json(fn)

	anno = pd.DataFrame(data['annotations'])
	anno['quadrant'] = list(map(aro_val_to_quads, anno['arousalValue'].tolist(), anno['valenceValue'].tolist()))

	users = pd.DataFrame(data['users'])

	df = pd.read_csv(csv, sep='\t', index_col=0)

	main(anno, df)



