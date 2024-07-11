from apify_client import ApifyClient
import requests
import pandas as pd
import os
# pd.set_option('display.max_colwidth', 100)  # Show full text in each column
from tqdm import tqdm
import uuid
tqdm.pandas()
import numpy as np
pd.set_option('display.max_colwidth', 150)
import autokeras as ak
import keras
import seaborn as sns
import re
import matplotlib.pyplot as plt


best_thresholds = {
     'O': 0.645,
     'C': 0.531,
     'E': 0.643,
     'A': 0.558,
     'N': 0.364}

mean_std_data = {
        '': ['Mean', 'Standard Deviation'],
        'postsCount': [1132.948864, 1579.897259],
        'followersCount': [5.416222e+06, 3.198482e+07],
        'followsCount': [1271.628788, 1296.381033],
        'avg_len_caption': [266.775577, 295.765561],
        'nb_hashtags': [273.638258, 463.213755],
        'nb_mentions': [0.414773, 1.656546],
        'duration': [800.691288, 842.021698],
        'frequency': [9.098853, 11.697160],
        'avg_emojis': [1.910339, 3.027327]
    }
mean_std_df = pd.DataFrame(mean_std_data).set_index('')
    

###########################################
def extract_username(url):
     try:
         return url.split("/")[-1]
     except Exception as e:
         print(f"Error processing URL '{url}': {e}")
     return None
#############################################
def url_image(url, user_folder,imgid=''):
     # Send a GET request to the image URL
     try:
       response = requests.get(url)
     except Exception as e:
       print(f"Error downloading image for URL {url}: {e}")
       return None
     # Check for successful response (status code 200)
     if response.status_code == 200:
         try:
             # Generate unique filename
             # image_files = os.listdir(user_folder)
             # index = len(image_files) + 1
             # Use first 8 characters of the UUID
             if(imgid==''):
                 imgid = str(uuid.uuid4())[:8]
             filename = os.path.join(user_folder, f"{imgid}.jpg")
             # filename = os.path.join(user_folder, f"{index}.png")

             # Save the image to file
             with open(filename, 'wb') as f:
                 f.write(response.content)

             return imgid
         except Exception as e:
             print(f"Error downloading image for URL {url}: {e}")
             return None
     else:
         print(
             f"Failed to download image for this URL {url} with Status code: {response.status_code}")
         return None
###########################################
def download_and_save_image(row):
     username = row['fullName']
     image_url = row['displayUrl']
     userurl = row['inputUrl']

     if username == '':
       username = extract_username(userurl)

     # Create folder for user if it doesn't exist

     user_folder = os.path.join(root_folder, username)
     if not os.path.exists(user_folder):
         os.makedirs(user_folder)

     # Download and save image
     image_filename = url_image(image_url, user_folder)
     return image_filename
 
###########################################
def download_and_save_pdp(url,username):
     # Create folder for user if it doesn't exist
     user_folder = os.path.join(root_folder, username)
     if not os.path.exists(user_folder):
         os.makedirs(user_folder)
     # Download and save image
     image_filename = url_image(url, user_folder,'pdp')
     return image_filename
###########################################
def extract_username(url):
     try:
         return url.split("/")[-1]
     except Exception as e:
         print(f"Error processing URL '{url}': {e}")
         return None
###########################################
def scraper(url):    
    apify_client = ApifyClient(APIToken)
    username = extract_username(url)
    if username is None:
       return("Invalid URL provided")
    print(username) 
    users = [username]
    resultsLimit = 2
    actor_collection_client = apify_client.actors()
    actor_collection_client.list().items
    post_actor = ""
    profile_actor = ""
    for actor in actor_collection_client.list().items:
      if actor['title'] == 'Instagram Post Scraper':
        post_actor = actor['id']
      if actor['title'] == 'Instagram Profile Scraper':
        profile_actor = actor['id']

    print(post_actor)
    print(profile_actor)
    # Prepare the Actor input
    run_input = {
        "username": users,
        "resultsLimit": resultsLimit,
    }
    # Run the Actor and wait for it to finish
    run = apify_client.actor(post_actor).call(run_input=run_input)
    # Fetch and print Actor results from the run's dataset (if there are any)
    desired_order = ['inputUrl', 'caption', 'displayUrl','timestamp']
    post_data = []
    print('post data')
    for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
        data = {}
        for k in desired_order:
          try:
            data[k] = item[k]
          except Exception as e:
            pass
        post_data.append(data)
    post_data = pd.DataFrame(post_data)
    print(post_data)
    # Prepare the Actor input
    run_input = {"usernames": users}
    # Run the Actor and wait for it to finish
    print('profile data')
    run = apify_client.actor(profile_actor).call(run_input=run_input)
    # Fetch and print Actor results from the run's dataset (if there are any)
    desired_order = ['inputUrl','fullName', 'followsCount','followersCount','biography','postsCount']
    profile_data = []
    for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
        data = {}
        for k in desired_order:
          try:
            pdp=item['profilePicUrlHD']
            data[k] = item[k]
          except Exception as e:
            pass
        # print(data)
        profile_data.append(data)
    profile_data = pd.DataFrame(profile_data)
    print(profile_data)
    df_merged = pd.merge(profile_data, post_data, on='inputUrl', how='outer')
    df_merged[df_merged['inputUrl'] == None]
    download_and_save_pdp(pdp,df_merged['fullName'].iloc[0])
    df_merged['image id'] =  df_merged.progress_apply(download_and_save_image, axis=1)
    # df_merged.drop(['inputUrl','displayUrl'],axis=1,inplace=True)
    print(df_merged)
    return df_merged

########################################### process ###########################################
def count_emojis(text):
    emoji_pattern = r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])'
    emojis = re.findall(emoji_pattern, text)
    return len(emojis)

def processdata(df):
    df.drop(['inputUrl'],axis=1,inplace=True)
    order = ['postsCount','followersCount','followsCount','caption','timestamp']
    df = df[order]
    df = df.groupby(['postsCount', 'followersCount','followsCount']).agg({
        'caption': list,
        'timestamp': list
    }).reset_index()

    df['dates'] = df['timestamp'].apply(lambda x: [pd.to_datetime(ts, errors='coerce') for ts in x])
    df['len_caption'] = [[len(x) for x in c] for c in df['caption']]
    df['avg_len_caption'] = df['len_caption'].apply(lambda x: np.average(x))
    df['nb_hashtags'] = df['caption'].apply(lambda x: sum([caption.count('#') for caption in x]))
    df['nb_mentions'] = df['caption'].apply(lambda x: sum([caption.count('@') for caption in x]))
    df['duration'] = df['dates'].apply(lambda x: (x[0] - x[-1]).days)
    df['frequency'] = df.apply(lambda row: row['duration'] / len(row['dates']), axis=1)
    df['nb_emoji']= [[count_emojis(x) for x in c] for c in df['caption']]
    df['avg_emojis'] = df['nb_emoji'].apply(lambda x: np.average(x) if len(x) > 0 else 0)

    df = df.drop(['timestamp','len_caption','nb_emoji','dates','caption'],axis=1)
    df = (df - mean_std_df.loc['Mean']) / mean_std_df.loc['Standard Deviation']
    print('data after processing',df)
    return df
    




def model(df):
    model = keras.models.load_model('/content/drive/MyDrive/PFA/best_model_num.keras')
    x = np.array(df)
    predictions = model(x)
    trait_keys = ['O', 'C', 'E', 'A', 'N']
    binary_predictions = {}
    prob_predictions = {}
    for idx, trait in enumerate(trait_keys):
        threshold = best_thresholds[trait]
        prob_predictions[trait] = np.squeeze(predictions[idx]).item()
        binary_predictions[trait] = (np.squeeze(predictions[idx]) > threshold).astype(int)
    print(binary_predictions,prob_predictions)
    return (binary_predictions,prob_predictions)
    
    plt.figure(figsize=(12, 5))
    prob_predictions = {k:(v*100 if v >0.05 else 5) for k,v in prob_predictions.items()}
    sns.barplot(x=list(prob_predictions.values()), y=list(prob_predictions.keys()), hue=list(prob_predictions.keys()), palette="Blues_d", dodge=False, legend=False)
    for index, value in enumerate(prob_predictions.values()):
        plt.text(value, index, f'{value:.2f}%', va='center')
    

    
    
    
def app(url):
    root_folder = 'D:\Desktop\scraper'
    APIToken = "apify_api_3VQat8MRztay2Aw2zih5IHHVJrsXEH0yCzL5"
    df_scrap=scraper(url)
    print('scrape done ----------------------------------')
    df=processdata(df_scrap)
    print ('process done -------',df)
    prediction=model(df)
    print ('prediction done -------',prediction)
    user_details = {
        "fullName": df_scrap['fullName'].iloc[0],
        "followsCount": df_scrap['followsCount'].iloc[0],
        "followersCount": df_scrap['followersCount'].iloc[0],
        "biography": df_scrap['biography'].iloc[0],
        "captions": df_scrap['caption'].tolist(),
        "image_ids": df_scrap['image id'].tolist(),
        "personality":prediction
    }
    return user_details 
     
    
    
    
    
    
    
    
    
root_folder = 'scraper'
APIToken = "apify_api_3VQat8MRztay2Aw2zih5IHHVJrsXEH0yCzL5"
def app(url):
    root_folder = 'scraper'
    APIToken = "apify_api_3VQat8MRztay2Aw2zih5IHHVJrsXEH0yCzL5"
    df_scrap=scraper(url)
    print('scrape done ----------------------------------')
    df=processdata(df_scrap)
    print ('process done -------',df)
    prediction=model(df)
    print ('prediction done -------',prediction)
    user_details = {
        "fullName": df_scrap['fullName'].iloc[0],
        "followsCount": df_scrap['followsCount'].iloc[0],
        "followersCount": df_scrap['followersCount'].iloc[0],
        "biography": df_scrap['biography'].iloc[0],
        "captions": df_scrap['caption'].tolist(),
        "image_ids": df_scrap['image id'].tolist(),
        "personality":prediction
    }
    return user_details 
    