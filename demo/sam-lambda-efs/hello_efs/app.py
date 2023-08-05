import json
import os
from pathlib import Path
import urllib
import boto3
import mrcfile
import numpy as np
import imageio
import shutil
import time
import base64

# You can reference EFS files by including your local mount path, and then
# treat them like any other file. Local invokes may not work with this, however,
# as the file/folders may not be present in the container.
ROOT_EFS_PATH = Path('/mnt/lambda/')
s3_client = boto3.client('s3')

def auto_contrast(image, t_mean=150.0/255.0, t_sd=40.0/255.0):
    image = (image - image.min()) / (image.max() - image.min())
    mean = image.mean()
    sq = image ** 2
    temp = sq.mean()
    sd = temp - mean ** 2
    sd = np.sqrt(sd)

    f = sd / t_sd

    black = mean - t_mean * f 
    white = mean + (1 - t_mean) * f
    
    new_image = np.clip(image, black, white)
    new_image = (new_image - black) / (white - black)
    return new_image

def convert_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def lambda_handler(event, context):

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    print(bucket, key)

    # downloading mrc to EFS ( mounted on an EC2 instance )
    mrc_name = Path(key)
    subroot_path = ROOT_EFS_PATH.joinpath(mrc_name.stem) # "/mnt/lambda/${mrc_stem}"
    # create subroot path
    if subroot_path.exists():
        shutil.rmtree(subroot_path, ignore_errors=True)
    os.makedirs(subroot_path, exist_ok=True)
    os.makedirs(subroot_path.joinpath('frames')) # "/mnt/lambda/${mrc_stem}/frames"
    print(subroot_path)
    
    # Parse mrc into {num_views} pngs
    base64_dict = {}
    s3_client.download_file(bucket, key, subroot_path.joinpath(f'{mrc_name.stem}.mrc'))
    mrc_data = mrcfile.open(subroot_path.joinpath(f'{mrc_name.stem}.mrc')).data.astype(np.float32)
    for i, frame in enumerate(mrc_data):
        frame = (auto_contrast(frame)*255).astype(np.uint8)
        frame_path = subroot_path.joinpath(f'frames/{mrc_name.stem}_{str(i).zfill(5)}.png')
        print(frame_path)
        imageio.imsave(frame_path, frame)
        # Convert png to base64
        base64_string = convert_image_to_base64(frame_path)
        base64_dict[f'{mrc_name.stem}_{str(i).zfill(5)}.png'] = base64_string
        # Upload pngs from EFS to S3
        bucket_key = f'{mrc_name.parent}/{mrc_name.stem}/{mrc_name.stem}_{str(i).zfill(5)}.png'
        s3_client.upload_file(frame_path, bucket, bucket_key)
    # Save base64 into mrc_name.json
    s3_client.put_object(Body=json.dumps(base64_dict, indent=4), Bucket=bucket, Key=f'{mrc_name.parent}/{mrc_name.stem}.json')

    # Update info.json in user's directory
    user_root_path = mrc_name.parents[2]
    proj_name = mrc_name.parents[1].stem
    info_json = user_root_path.joinpath('info.json')

    response = s3_client.get_object(Bucket=bucket, Key=str(info_json))
    content = response['Body'].read().decode('utf-8')
    info_dict = json.loads(content)

    info_dict[proj_name]['last_modify_time'] = str(int(time.time()))
    info_dict[proj_name]['thumbnail'] = base64_dict[next(iter(base64_dict))]
    info_dict[proj_name]['progress'] = 1
    print(info_dict)

    json_str = json.dumps(info_dict, indent=4)
    bytes_data = json_str.encode('utf-8')
    response = s3_client.put_object(Bucket=bucket, Key=str(info_json), Body=bytes_data)
