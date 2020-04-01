from io import BytesIO
import boto3
import numpy as np
from PIL import Image
from botocore import UNSIGNED, exceptions
from botocore.config import Config
import cv2

class S3Util:
    @staticmethod
    def fetch_image_unsigned(bucket,key):
        try:
            s3=boto3.client('s3',config=Config(signature_version=UNSIGNED))
            bytes_io=BytesIO()
            s3.download_fileobj(bucket,key,bytes_io)
            im=Image.open(bytes_io)
            img = np.array(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                #print("The object does not exist.")
                return None
            else:
                return None

    @staticmethod
    def fetch_image(bucket,key):
        s3=boto3.resource('s3',)
        bucket=s3.Bucket(bucket)
        object=bucket.Object(key)
        response=object.get()
        file_stream=response['Body']
        im=Image.open(file_stream)
        return np.array(im)

    @staticmethod
    def write_image(img_array,bucket,key, config):
        s3=boto3.resource('s3',config)
        bucket=s3.Bucket(bucket)
        object=bucket.Object(key)
        file_stream=BytesIO()
        im=Image.fromarray(img_array)
        im.save(file_stream,format='jpeg')
        object.put(Body=file_stream.getvalue())
