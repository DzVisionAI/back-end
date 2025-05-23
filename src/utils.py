import os
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta

_gcs_client = None

def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            _gcs_client = storage.Client(credentials=credentials)
        else:
            _gcs_client = storage.Client()  # fallback to default creds
    return _gcs_client

def generate_gcs_signed_url(blob_name, expiration=3600):
    """
    Generate a signed URL for a GCS object (blob_name) valid for 'expiration' seconds.
    """
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    if not bucket_name:
        raise Exception('GCS_BUCKET_NAME environment variable not set')
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(expiration=timedelta(seconds=expiration))
    return url

def upload_to_gcs(local_path, destination_blob_name, signed_url_expiration=3600):
    """
    Uploads a file to GCS and returns a dict with the blob URL and a signed URL.
    """
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    if not bucket_name:
        raise Exception('GCS_BUCKET_NAME environment variable not set')
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    blob_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
    signed_url = blob.generate_signed_url(expiration=timedelta(seconds=signed_url_expiration))
    return {"blob_url": blob_url, "signed_url": signed_url}

def test_gcs_upload(local_image_path):
    """
    Uploads a test image to GCS and prints the blob and signed URLs.
    """
    test_blob_name = f"test_uploads/{os.path.basename(local_image_path)}"
    result = upload_to_gcs(local_image_path, test_blob_name)
    print(f"Test image uploaded to: {result['blob_url']}")
    print(f"Signed URL: {result['signed_url']}")
    return result
