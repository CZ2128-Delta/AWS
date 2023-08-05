##########################
# Set up the environment #
##########################

import boto3
from sagemaker import get_execution_role

sm_client = boto3.client(service_name="sagemaker")
runtime_sm_client = boto3.client(service_name="sagemaker-runtime")

account_id = boto3.client("sts").get_caller_identity()["Account"]
region = boto3.Session().region_name

bucket = "sagemaker-{}-{}".format(region, account_id)
prefix = "demo-multimodel-endpoint"

role = "arn:aws:iam::133917102535:role/SageMaker-cz2128" # get_execution_role()

################################
# Upload model artifacts to S3 #
################################

import mxnet as mx
import os
import tarfile

model_path = "http://data.mxnet.io/models/imagenet/"

# Download ResNet 18

mx.test_utils.download(
    model_path + "resnet/18-layers/resnet-18-0000.params", None, "data/resnet_18"
)
mx.test_utils.download(
    model_path + "resnet/18-layers/resnet-18-symbol.json", None, "data/resnet_18"
)
mx.test_utils.download(model_path + "synset.txt", None, "data/resnet_18")

with open("data/resnet_18/resnet-18-shapes.json", "w") as file:
    file.write('[{"shape": [1, 3, 224, 224], "name": "data"}]')

with tarfile.open("data/resnet_18.tar.gz", "w:gz") as tar:
    tar.add("data/resnet_18", arcname=".")

# Download ResNet 152

mx.test_utils.download(
    model_path + "resnet/152-layers/resnet-152-0000.params", None, "data/resnet_152"
)
mx.test_utils.download(
    model_path + "resnet/152-layers/resnet-152-symbol.json", None, "data/resnet_152"
)
mx.test_utils.download(model_path + "synset.txt", None, "data/resnet_152")

with open("data/resnet_152/resnet-152-shapes.json", "w") as file:
    file.write('[{"shape": [1, 3, 224, 224], "name": "data"}]')

with tarfile.open("data/resnet_152.tar.gz", "w:gz") as tar:
    tar.add("data/resnet_152", arcname=".")

# Upload

from botocore.client import ClientError
import os

s3 = boto3.resource("s3")
try:
    s3.meta.client.head_bucket(Bucket=bucket)
except ClientError:
    s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})

models = {"resnet_18.tar.gz", "resnet_152.tar.gz"}

for model in models:
    key = os.path.join(prefix, model)
    with open("data/" + model, "rb") as file_obj:
        s3.Bucket(bucket).Object(key).upload_fileobj(file_obj)

##############################
# Import models into hosting #
##############################

from time import gmtime, strftime

model_name = "DEMO-MultiModelModel" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model_url = "https://s3-{}.amazonaws.com/{}/{}/".format(region, bucket, prefix)
container = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(
    account_id, region, "demo-sagemaker-multimodel"
)

print("Model name: " + model_name)
print("Model data Url: " + model_url)
print("Container image: " + container)

container = {"Image": container, "ModelDataUrl": model_url, "Mode": "MultiModel"}

create_model_response = sm_client.create_model(
    ModelName=model_name, ExecutionRoleArn=role, Containers=[container]
)

print("Model Arn: " + create_model_response["ModelArn"])

#################################
# Create endpoint configuration #
#################################

endpoint_config_name = "DEMO-MultiModelEndpointConfig-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Endpoint config name: " + endpoint_config_name)

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m5.xlarge",
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 1,
            "ModelName": model_name,
            "VariantName": "AllTraffic",
        }
    ],
)

print("Endpoint config Arn: " + create_endpoint_config_response["EndpointConfigArn"])

###################
# Create endpoint #
###################

import time

endpoint_name = "DEMO-MultiModelEndpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Endpoint name: " + endpoint_name)

create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])

resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
print("Endpoint Status: " + status)

print("Waiting for {} endpoint to be in service...".format(endpoint_name))
waiter = sm_client.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=endpoint_name)

#################
# Invoke models #
#################

import json

fname = mx.test_utils.download(
    "https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true",
    "cat.jpg",
)

with open(fname, "rb") as f:
    payload = f.read()

# Invoke ResNet 18

response = runtime_sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/x-image",
    TargetModel="resnet_18.tar.gz",  # this is the rest of the S3 path where the model artifacts are located
    Body=payload,
)

print(*json.loads(response["Body"].read()), sep="\n")

# Invoke ResNet 152

response = runtime_sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/x-image",
    TargetModel="resnet_152.tar.gz",
    Body=payload,
)

print(*json.loads(response["Body"].read()), sep="\n")
