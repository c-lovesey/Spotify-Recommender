import csv
import difflib
import glob
import itertools
import json
import os
import re
import shutil
import statistics
import sys
import threading
import tkinter as tk
import tkinter.messagebox as msgbox
import webbrowser
from collections import defaultdict
from json.decoder import JSONDecodeError
from pprint import pprint

import datapane as dp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pydot
import requests
import scipy
import seaborn as sns
import spotipy
import spotipy.util as util
from PIL import Image, ImageTk
from scipy.spatial.distance import cdist
from sklearn import decomposition, preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth


def main():
    parent = os.path.dirname(__file__)
    dir  = "data"
    csv = '.csv'
    path = os.path.join(parent, dir)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    print(path)
    SPOTIPY_CLIENT_ID='a96c59cf154246c4911fc6934d7249cb'
    SPOTIPY_CLIENT_SECRET='61de126703ce46cbba72a87ba7e9d77e'
    SPOTIPY_REDIRECT_URI='https://www.google.co.uk/'
    scope = 'user-read-recently-played user-read-playback-state app-remote-control user-read-private playlist-read-private user-modify-playback-state playlist-modify-public user-modify-playback-state playlist-modify-private user-read-playback-position user-library-read'
    token = Authorisation(SPOTIPY_CLIENT_ID,SPOTIPY_CLIENT_SECRET,SPOTIPY_REDIRECT_URI,scope)
    #username = '21bgy75cvahxgq6763og2ppyi'


    spotifyObject = spotipy.Spotify(auth=token, requests_timeout=10, retries=10)
    
    # Current track information
    track = spotifyObject.current_user_playing_track()
    #artist = track['item']['artists'][0]['name']
    #track = track['item']['name']

   

    
    parent = os.path.dirname(__file__)
    dir  = "data"
    csv = '.csv'
    user = spotifyObject.current_user()
    displayName = user['display_name']
    #print(displayName)
    followers = user['followers']['total']
    username = user['id']
    names = []
    #df = json.loads(library)
    append = False
    #Create_Playlist(username,spotifyObject)
    Playlists = getPlaylistIDs(username,spotifyObject,limit = 50)
    #print(Playlists)
    #trackids = getTrackIDs(username,Playlists,spotifyObject)
    tracks = pd.DataFrame()
    upsf = pd.DataFrame()
    count = 0
    csv = '.csv'
    
    for item in Playlists:   
        name = spotifyObject.playlist(item)
        names.append(name['name'])
        path = os.path.join(parent, dir , str(item) + csv).replace("\\","/")
        print(path)
        temp = []
        song_data = {}
        try:
            upsf = pd.read_csv(path,index_col = 0)
        except IOError:
            print("error")
            upsf =pl(item,upsf,spotifyObject,token,username,append) 
    print("Choose a playlist to reccomend for (use numbers)")
    ii = 0
    for item in names:
        
        print(ii,item)
        ii = ii +1
    while True:
        choice = input("Input Number:")
        choice = int(choice)

        
        try:

            name = names[choice]
            print(name)
            name = Playlists[choice]
            break
        except ValueError:
            print("wrong number")
        
    
    
    spotify_df,genre_data,data_by_year  = LoadDataset(spotifyObject)
    spotify_data1 = spotify_df

    spotify_data = spotify_data1[['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo','id']].copy()
    print(spotify_data)
    
    #spotify_df = spotify_df.drop(['artists','release_date','explicit','instrumentalness','key','name','release_date'],axis = 1)
    #spotify_df.drop(['year','artists','name','artists_upd_v1','artists_upd_v2','artists_upd','release_date'])
    #print("do drop")
    #print(spotify_df)
    # filename = os.path.join(path, 'TracksDataset.csv')
    # upsf.to_csv(filename,sep = ',')

   #create cluster pipeline
    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10, n_jobs=-1))])
    #get the genredata
    X = genre_data.select_dtypes(np.number)
    #fir the data to 
    cluster_pipeline.fit(X)
    genre_data['cluster'] = cluster_pipeline.predict(X)

    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    # projection['genres'] = genre_data['genres']
    # projection['cluster'] = genre_data['cluster']
    

    # fig = px.scatter( projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
    # fig.show()

   
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=2, n_jobs=4))],verbose=True)
    X = spotify_data.select_dtypes(np.number)
    number_cols = list(X.columns)
    song_cluster_pipeline.fit(X)
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    song_cluster_labels = song_cluster_pipeline.predict(X)
    spotify_data['cluster_label'] = song_cluster_labels

    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)

    parent = os.path.dirname(__file__)
    child  = "data"
    path = os.path.join(parent, child)
    filename = os.path.join(path, 'data_o.csv')

    n_songs = 50

    
    playlisti = name
    
    playlist = playlisti
    csv = '.csv'

    path = os.path.join(parent, dir , str(playlist) + csv).replace("\\","/")
    print(path)
    temp = []
    song_data = {}
    try:
        data = pd.read_csv(path)
    except IOError:
        print("error")
   # print(upsf)
    data = pd.read_csv(path)


    #print(data)
    pli = data
    print(pli)
    song_vectors = []
    for item in pli['id']:
        #print(item)
        metadata_cols = ['name', 'year', 'artists']
        song_vectors = [] 
        #pprint(spotify_data)
        meta = spotifyObject.track(item)
        id = meta['album']['id']
        print(id)
        if id == None:
            continue
        features = spotifyObject.audio_features(item)[0] 

        uri = str(meta['album']['release_date'])
        year = uri.split('-')
        year = year[0]
        # features
        name = str(meta['album']['name'])
        song_data = {}
        explicit  = int(meta['explicit'])
        duration = meta['duration_ms']
        key = features['key']
        popularity = meta['popularity']
        acousticness = features['acousticness']
        danceability = features['danceability']
        energy = features['energy']
        instrumentalness = features['instrumentalness']
        liveness = features['liveness']
        loudness = features['loudness']
        speechiness = features['speechiness']
        tempo = features['tempo']
        mode = features['mode']
        time_signature = features['time_signature']
        valence =features['valence']
        duration_ms = features['duration_ms']

        song_data['year'] = year
        song_data['explicit'] = explicit
        song_data['key'] = key
        song_data['duration_ms'] = duration
        song_data['popularity'] = popularity
        song_data['acousticness'] = acousticness
        song_data['danceability'] = danceability
        song_data['energy'] = energy
        song_data['instrumentalness'] = instrumentalness
        song_data['liveness'] = liveness
        song_data['loudness'] = loudness
        song_data['speechiness'] = speechiness
        song_data['tempo'] = tempo
        song_data['mode'] = mode
        song_data['time_signature'] = time_signature
        song_data['valence'] = valence
        song_data['duration_ms'] = duration_ms

        song_vectors.append(song_data)   
    song_data = pd.DataFrame(data = song_vectors, columns = number_cols)
    
    song_matrix = np.array(song_data).astype(np.float)
    song_center = np.mean(song_matrix, axis=0)

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    pprint(rec_songs)
    #rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    rec_songs = pd.DataFrame(rec_songs)
    pprint(rec_songs)
    index = rec_songs.index
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo','id']
    temp =[]
    pprint(index) 
    upsf = pd.read_csv(path,index_col = 0)
    print(pli)
    for i in index: 
        print(i)
        out = spotify_data.iloc[i] 
        print(out)
        temp.append(out)
    song_data = pd.DataFrame(data = temp, columns = number_cols)
    print(song_data)
    spotifyObject.playlist_add_items(playlisti, song_data['id'])
    append = True
    
    upsf =pl(playlisti,upsf,spotifyObject,token,username,append)
    upsf.to_csv(path,sep = ',')
    return    


def rf(tracks, dataset,length):
    length = int(len(length))
    df = pd.DataFrame(tracks)
    Xdanceability = df['danceability']
    Xenergy = df['energy']
    Xkey = df['key']
    Xloudness = df['loudness']
    Xmode = df['mode']
    Xspeechiness = df['speechiness']
    Xacousticness = df['acousticness']
    Xinstrumentalness = df['instrumentalness']
    Xliveness = df['liveness']
    Xvalence = df['valence']
    Xtempo = df['tempo']
    ds = pd.DataFrame(dataset)
    Ydanceability = ds['danceability']
    Yenergy = ds['energy']
    Ykey = ds['key']
    Yloudness = ds['loudness']
    Ymode = ds['mode']
    Yspeechiness = ds['speechiness']
    Yacousticness = ds['acousticness']
    Yinstrumentalness = ds['instrumentalness']
    Yliveness = ds['liveness']
    Yvalence = ds['valence']
    Ytempo = ds['tempo']                

    df = pd.DataFrame(columns= [Xdanceability,Xenergy, Xkey,Xloudness, Xmode,Xspeechiness, Xacousticness, Xinstrumentalness,Xliveness,Xvalence,Xtempo])
    df = df.dropna(axis = 0, how = 'any')
    kmeans = KMeans(n_clusters=11)
    ds = pd.DataFrame(columns= [Ydanceability,Yenergy, Ykey,Yloudness, Ymode,Yspeechiness, Yacousticness, Yinstrumentalness,Yliveness,Yvalence,Ytempo])
    ds = df.dropna(axis = 0, how = 'any')
    #print(df,'\n',ds)
    y = kmeans.fit([Ydanceability,Yenergy, Ykey,Yloudness, Ymode,Yspeechiness, Yacousticness, Yinstrumentalness,Yliveness,Yvalence,Ytempo])
    df['Cluster'] = y
    print(df['Cluster'])

    #t = fit(X, kmeans, 1)
    #recommendations = predict(t, Y)
    #output = recommend(recommendations, metadata, Y)


    return 

def predict(t, Y):
    y_pred = t[1].predict(Y)
    mode = pd.Series(y_pred).mode()
    return t[0][t[0]['label'] == mode.loc[0]]

def recommend(recommendations, meta, Y):
    dat = []
    for i in Y['id']:
        dat.append(i)
    genre_mode = meta.loc[dat]['genre'].mode()
    artist_mode = meta.loc[dat]['artist_name'].mode()
    return meta[meta['genre'] == genre_mode.iloc[0]], meta[meta['artist_name'] == artist_mode.iloc[0]], meta.loc[recommendations['track_id']]


def playlist_weight(playlist):
    
    df = pd.DataFrame(playlist)
    out = pd.DataFrame()
    out2 = pd.DataFrame()
    plid = playlist['playlist_id'].drop_duplicates()
    #out = []
    for item in plid:
        currentitem = df.loc[df.playlist_id==item]

        playlistid = currentitem['playlist_id']
        trackid = currentitem['uri']
        danceability = currentitem['danceability']
        energy = currentitem['energy']
        key = currentitem['key']
        loudness = currentitem['loudness']
        mode = currentitem['mode']
        speechiness = currentitem['speechiness']
        acousticness = currentitem['acousticness']
        instrumentalness = currentitem['instrumentalness']
        liveness = currentitem['liveness']
        valence = currentitem['valence']
        tempo = currentitem['tempo']
        currentitem['in_playlist'] = True
        in_playlist = currentitem['in_playlist']
        temp = pd.DataFrame({'playlist_id': playlistid,'trackid':trackid,'danceability': danceability,'energy': energy,'key': key, 
                            'loudness': loudness,'mode': mode,'speechiness': speechiness,'acousticness': acousticness,
                            'instrumentalness': instrumentalness,'liveness': liveness,'valence': valence,'tempo':tempo,'in_playlist':in_playlist})
        weigh = weight(temp)
        out = out.append(temp)
        out2 = out2.append(weigh)

    return out,out2

def train(out):
    

    return
def weight(df):

    playlistid = df['playlist_id']

    danceability = df['danceability'].mean()

    #danceability = statistics.mean(danceability)
    energy = df['energy'].mean()
    #energy = statistics.mean(energy)
    key = df['key'].mean()
   #key = statistics.mean(key)
    loudness = df['loudness'].mean()
    #loudness = loudness.mean(loudness)
    mode = df['mode'].mean()
    #mode = statistics.mean(mode)
    speechiness = df['speechiness'].mean()
    #speechiness = statistics.mean(speechiness)
    acousticness = df['acousticness'].mean()
    #acousticness = statistics.mean(acousticness)
    instrumentalness = df['instrumentalness'].mean()
    #instrumentalness = statistics.mean(instrumentalness)
    liveness = df['liveness'].mean()
    #liveness = statistics.mean(liveness)
    valence = df['valence'].mean()
    #valence = statistics.mean(valence)
    tempo = df['tempo'].mean()
    #tempo = statistics.mean(danceability)
    temp = pd.DataFrame({'playlist_id': playlistid,'danceability': danceability,'energy': energy,'key': key, 
                        'loudness': loudness,'mode': mode,'speechiness': speechiness,'acousticness': acousticness,
                        'instrumentalness': instrumentalness,'liveness': liveness,'valence': valence,'tempo':tempo})
    temp = temp.drop_duplicates()
    return temp
def stats(sp):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data/Listening_history.csv')
    ammount = {}
    record = []
    ftime = []
    out = []

    stats = sp.current_user_recently_played(limit = 50)
    out_file = open("temp.json","w")
    out_file.write(json.dumps(stats, sort_keys=True))

    with open('temp.json','r') as f:
        data = json.loads(f.read())
    # Flatten data
    df_nested_list = pd.json_normalize(data, record_path =['items'])
    #print(df_nested_list)
    track = df_nested_list['track.uri']
    time = df_nested_list['played_at']
    for i in range(len(df_nested_list['track.uri'])):
        id = str(track[i])
        temp = id.split(":",2)
        out.append(temp[2])
    #out = out * 2
    for ii in range(len(out)):
        ammount[out[ii]] = out.count(out[ii])
    keys = list(ammount.keys())

    values = list(ammount.values())
    my_dict_3 = dict(zip(keys, zip(values, time)))
    output = pd.DataFrame(my_dict_3)
    output.to_csv(filename)

    #textfile.close()
    # textfile = open("Listening_history.txt", "a")
    # for iii in range(len(ammount)):
    #     textfile.write(ammount[] + ":" + element.value() + ",")
        
    #out.to_csv("Listening_history.csv",sep = ',')
    #textfile.close()
    return output
def get_playlist_track_id(playlist_id,sp):
    playlist_ids = 'spotify:playlist:' + str(playlist_id)
    track_names = []
    track_id = []
    names = []
    offset = 0
    while True:
        response = sp.playlist_items(playlist_ids,offset=offset)
        offset = offset + 50
        
        for item in response['items']:
            track = item['track']
            name  = track['name']
            id = track['id']
            track_id.append(id)
            track_names.append(name)

        if len(response['items']) == 0:
            break
  
    


    
    temp = pd.DataFrame(track_id, columns = ["id"])
    temp["name"] = track_names
 


    return temp

def pl(sourcePlaylistID,f,sp,token,username,append):
    SPOTIPY_CLIENT_ID='a96c59cf154246c4911fc6934d7249cb'
    SPOTIPY_CLIENT_SECRET='61de126703ce46cbba72a87ba7e9d77e'
    SPOTIPY_REDIRECT_URI='https://www.google.co.uk/'
    scope = 'user-read-recently-played user-read-playback-state app-remote-control user-read-private playlist-read-private user-modify-playback-state playlist-modify-public user-modify-playback-state playlist-modify-private user-read-playback-position user-library-read'
    col =['explicit','duration_ms','key','popularity','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','mode','time_signature','valence','duration_ms','id','playlist_id']
    print(col)
    parent = os.path.dirname(__file__)
     

    track_ids = []
    track_names = []

    ids = []
    remove = 0
    temp = []
    dir  = "data"
    path = os.path.join(parent, dir)
    filename = os.path.join(path, 'data_o.csv')
    if append == True:
        spotify_df = pd.read_csv(filename)
    #print("Analysing Playlists")
    x = isinstance(sourcePlaylistID, str)
    print(sourcePlaylistID)
    playlist_df = f

    try:
        if x == True:
            playlist = sourcePlaylistID
            csv = '.csv'
            path = os.path.join(parent, dir , str(playlist) + csv)
            child = 'data/' + str(playlist) + '.csv'
            path = os.path.join(parent, child)
            temp = []
            plidlist = []
            data = pd.DataFrame()
            playlistid = []
            song_data = {}
            if append == True:
                playlist_df = pd.read_csv(path)
            else:
                print("x")
                user = sp.current_user()
                username = user['id']
                tracks = getTrackIDs(username,playlist,sp)

                ids = list(tracks)

                #features = pd.DataFrame(data = None,columns = col)
                for i in range(0,len(ids)):
                    
                    if ids[i] != None:
                        track_ids = ids[i]
                    else:
                        continue
                   # print(track_ids)
                    meta = sp.track(track_ids)
                    
                    audio_features  = sp.audio_features(track_ids)

                    explicit  = int(meta['explicit'])
                    
                    key = audio_features[0]['key']
                    popularity = meta['popularity']
                    id = meta['id']
                    
                    acousticness = audio_features[0]['acousticness']
                    danceability = audio_features[0]['danceability']
                    energy = audio_features[0]['energy']
                    instrumentalness = audio_features[0]['instrumentalness']
                    liveness = audio_features[0]['liveness']
                    loudness = audio_features[0]['loudness']
                    speechiness = audio_features[0]['speechiness']
                    tempo = audio_features[0]['tempo']
                    mode = audio_features[0]['mode']
                    time_signature = audio_features[0]['time_signature']
                    valence =audio_features[0]['valence']
                    duration = audio_features[0]['duration_ms']
                    uri = str(meta['album']['release_date'])
                    year = uri.split('-')
                    year = year[0]
                    song_data['year'] = year
                    song_data['explicit'] = explicit
                    song_data['key'] = key
                    song_data['duration_ms'] = duration
                    song_data['popularity'] = popularity
                    song_data['acousticness'] = acousticness
                    song_data['danceability'] = danceability
                    song_data['energy'] = energy
                    song_data['instrumentalness'] = instrumentalness
                    song_data['liveness'] = liveness
                    song_data['loudness'] = loudness
                    song_data['speechiness'] = speechiness
                    song_data['tempo'] = tempo
                    song_data['mode'] = mode
                    song_data['time_signature'] = time_signature
                    song_data['valence'] = valence
                    song_data['id']=str(id)
                    song_data['playlist_id']=str(playlist) 

                    
                    data = data.append(song_data, ignore_index=True)
                pprint(data)
                features = pd.DataFrame(data = data,columns = col)
                
                #song_data = pd.DataFrame(data = features, columns = col)
            
                playlist_df = features
                print(playlist_df)
                playlist_df.to_csv(path,sep = ',')
            #playlist_df = playlist_df.concat([playlist_df,temp])
            #print(playlist_df)  
        else:
            print("y")
            for item in range(len(sourcePlaylistID)):
                plidlist = []
                playlistid = []
                                    
                track_ids = []
                track_names = []
                
                playlist = sourcePlaylistID[item]
                #print(playlist)
                user = sp.current_user()
                username = user['id']
                tracks = getTrackIDs(username,playlist,sp)
                ids = list(tracks)
                for i in range(len(ids)):
                    if ids[i] != None:
                        track_ids.append(ids[i])
                        plidlist.append(username)
                        playlistid.append(playlist)
                
            
                csv = '.csv'
                path = os.path.join(parent, dir , str(playlist) + csv)
                child = 'data/' + str(playlist) + '.csv'
                path = os.path.join(parent, child)
                df = pd.DataFrame(data = None, columns = col)
                if append == True:
                    playlist_df = pd.read_csv(path)

                

                
                ids = []
                temp = []
                song_data = {}
                
                data = pd.DataFrame()
                
                ids = list(tracks)
                for i in range(0,len(ids)):
                    
                    if ids[i] != None:
                        track_ids = ids[i]
                    else:
                        continue
                   # print(track_ids)
                    meta = sp.track(track_ids)
                    
                    audio_features  = sp.audio_features(track_ids)

                    explicit  = int(meta['explicit'])
                    
                    key = audio_features[0]['key']
                    popularity = meta['popularity']
                    id = meta['id']
                    
                    acousticness = audio_features[0]['acousticness']
                    danceability = audio_features[0]['danceability']
                    energy = audio_features[0]['energy']
                    instrumentalness = audio_features[0]['instrumentalness']
                    liveness = audio_features[0]['liveness']
                    loudness = audio_features[0]['loudness']
                    speechiness = audio_features[0]['speechiness']
                    tempo = audio_features[0]['tempo']
                    mode = audio_features[0]['mode']
                    time_signature = audio_features[0]['time_signature']
                    valence =audio_features[0]['valence']
                    duration = audio_features[0]['duration_ms']
                    uri = str(meta['album']['release_date'])
                    year = uri.split('-')
                    year = year[0]
                    song_data['year'] = year
                    song_data['explicit'] = explicit
                    song_data['key'] = key
                    song_data['duration_ms'] = duration
                    song_data['popularity'] = popularity
                    song_data['acousticness'] = acousticness
                    song_data['danceability'] = danceability
                    song_data['energy'] = energy
                    song_data['instrumentalness'] = instrumentalness
                    song_data['liveness'] = liveness
                    song_data['loudness'] = loudness
                    song_data['speechiness'] = speechiness
                    song_data['tempo'] = tempo
                    song_data['mode'] = mode
                    song_data['time_signature'] = time_signature
                    song_data['valence'] = valence
                    song_data['id']=str(id)
                    song_data['playlist_id']=str(playlist) 

                    
                    data = data.append(song_data, ignore_index=True)
                pprint(data)
                features = pd.DataFrame(data = data,columns = col)
                
                
            
                playlist_df = features
                print(playlist_df)
                playlist_df.to_csv(path,sep = ',')

                
            
    except spotipy.client.SpotifyException:
        token = Authorisation(SPOTIPY_CLIENT_ID,SPOTIPY_CLIENT_SECRET,SPOTIPY_REDIRECT_URI,scope)
        sp = spotipy.Spotify(auth=token)
    
    
   
    print(playlist_df)
    


    return playlist_df

def Authorisation(SPOTIPY_CLIENT_ID,SPOTIPY_CLIENT_SECRET,SPOTIPY_REDIRECT_URI,scope):
    
    token = util.prompt_for_user_token(
                            client_id=SPOTIPY_CLIENT_ID,
                            client_secret=SPOTIPY_CLIENT_SECRET,
                            redirect_uri=SPOTIPY_REDIRECT_URI,
                            scope = scope)

    return token

def getTrackIDs(user, playlist_id,spotifyObject):
    #print(playlist_id)
    #print("this da id ^")
    ids = pd.DataFrame()
    count = 0
    #ids = []
    offset = 0
    value = []
    while True:
        temp = pd.DataFrame()
        response = spotifyObject.playlist_items(playlist_id,
                                offset=offset,
                                fields='items.track.id,total',
                                additional_types=['track'])

        if len(response['items']) == 0:
            break
        offset = offset + len(response['items'])
        for i in response['items']:
            id = str(i)
            id = id.replace("}","")
            id = id.replace("'","")
            temp = id.split(": ",2)
                    
            value.append(temp[2])
    temp["playlist_id"] = value
    ids = ids.append(temp)
    idsf = ids['playlist_id'].tolist()


    return idsf
   

def getNames(spotifyObject):
    name = {}
    list_photo = {}
    for i in spotifyObject.current_user_playlists()['items']:

        name[i['name']] = i['uri'].split(':')[4]
        

    return name
def getPlaylistIDs(user,spotifyObject,limit):
    #get playlists
    library = spotifyObject.current_user_playlists(limit=limit)
    #set local variables
    ids = []
    playlist = library
    for item in playlist['items']:
        #loops through each playlist in list
        key = []
        value = []
        #cleaning the data to get the id of each playlist
        libdata = str(item)           
        libdata = libdata.replace("{","")
        libdata = libdata.replace("{","")
        libdata = libdata.replace("}","")
        libdata = libdata.replace("[","")
        libdata = libdata.replace("]","")
        libdata = libdata.replace("'","")
        libdata = libdata.split(", ")
        for i in range(len(libdata)):
            txt = str(libdata[i])
            temp = txt.split(": ",1)
            
            if len(temp) == 2:
                key.append(temp[0])
                value.append(temp[1])
        res = {}
        res = dict(zip(key, value))
        #saving ids as a list that can be itereated through
        temp = res['uri']
        temp = temp.split(":",2)
        ids.append(temp[2])
    
    return ids
# def Convert(items):
    #print(len(items))
    cv = []
    txt = []
    temp = [10]
    key = []
    item = []
    res = {}
    items = items['items']

    libdata = str(items)           
    libdata = libdata.replace("{","")
    libdata = libdata.replace("{","")
    libdata = libdata.replace("}","")
    libdata = libdata.replace("[","")
    libdata = libdata.replace("]","")
    libdata = libdata.replace("'","")
    libdata = libdata.split(", ")


    for i in range(len(libdata)):
        txt = str(libdata[i])
        temp = txt.split(": ",1)
        print(temp)
        if len(temp) == 2:


            
            cv.append(temp[0])
            key = key + cv
            cv.append(temp[1])
            item = item + cv
    res = {i : {temp[0]:temp[1]}}


              
    res = dict(zip(key, item))



    return res  
def search(spotifyObject):
    searchitem = input("Search: ")
    result = spotifyObject.search(searchitem)
    return result
def Create_Playlist(user, spotifyObject):
    name = input("Playlist Name: ")
    spotifyObject.user_playlist_create(user, name, public = False, description = "")

def getTrackFeatures(id,spotifyObject):
    #print(id)
    temp = []
    x = isinstance(id, str)
    if x == True:
        item = id
        meta = spotifyObject.track(item)
        features = spotifyObject.audio_features(item)

        # meta
        name = meta['name']
        artist = meta['album']['artists'][0]['name']
        length = meta['duration_ms']
        popularity = meta['popularity']
        uri = str(meta['album']['release_date'])

        year = uri.split('-')
        year = year[0]
        # features
        id  = features[0]['id']
        acousticness = features[0]['acousticness']
        danceability = features[0]['danceability']
        energy = features[0]['energy']
        instrumentalness = features[0]['instrumentalness']
        liveness = features[0]['liveness']
        loudness = features[0]['loudness']
        speechiness = features[0]['speechiness']
        tempo = features[0]['tempo']
        mode = features[0]['mode']
        time_signature = features[0]['time_signature']
        valence =features[0]['valence']
        duration_ms = features[0]['duration_ms']
        track = [popularity, danceability, acousticness, danceability, energy, liveness, loudness,mode, speechiness, tempo,valence,year,id]
        temp.append(track)
        return temp
    else:
        for item in (id):
            print(item)
            meta = spotifyObject.track(item)
            features = spotifyObject.audio_features(item)

            # meta
            name = meta['name']
            artist = meta['album']['artists'][0]['name']
            length = meta['duration_ms']
            popularity = meta['popularity']
            uri = str(meta['album']['release_date'])

            year = uri.split('-')
            year = year[0]
            # features
            id  = features[0]['id']
            acousticness = features[0]['acousticness']
            danceability = features[0]['danceability']
            energy = features[0]['energy']
            instrumentalness = features[0]['instrumentalness']
            liveness = features[0]['liveness']
            loudness = features[0]['loudness']
            speechiness = features[0]['speechiness']
            tempo = features[0]['tempo']
            mode = features[0]['mode']
            time_signature = features[0]['time_signature']
            valence =features[0]['valence']
            duration_ms = features[0]['duration_ms']
            track = [popularity, danceability, acousticness, danceability, energy, liveness, loudness,mode, speechiness, tempo,valence,year,id]
            temp.append(track)
    return temp

def LoadDataset(spotifyObject):
    parent = os.path.dirname(__file__)
    child  = "data"
    path = os.path.join(parent, child)
    filename = os.path.join(path, 'data_o.csv')
    spotify_df = pd.read_csv(filename)
    filename = os.path.join(path, 'data_by_genres_o.csv')
    genre_data = pd.read_csv(filename)
    filename = os.path.join(path, 'data_by_year_o.csv')
    data_by_year = pd.read_csv(filename)


    #print(spotify_df)
    spotify_df.drop(['name'],axis = 1)

    #filename = os.path.join(path, 'tracks.csv')
    #spotify_df.to_csv(filename,sep = ',')

    return spotify_df,genre_data,data_by_year
    






if __name__ == '__main__':
    main()
