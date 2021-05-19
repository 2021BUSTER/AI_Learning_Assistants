import boto3 
from botocore.exceptions import NoCredentialsError

ACCESS_KEY = 'AKIAX4F3GCZSM6TRSRXS'
SECRET_KEY = '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'

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    try:
        print(s3.upload_file(local_file, bucket, s3_file))
        return True
    except FileNotFoundError:
        return False
    except NoCredentialsError:
        return False


uploaded = upload_to_aws('savefig_200dpi0.png', 'bucektmin', 'testimage.png')


